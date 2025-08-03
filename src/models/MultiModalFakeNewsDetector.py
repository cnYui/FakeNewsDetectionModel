import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
from transformers import (
    DistilBertTokenizer, DistilBertModel,
    CLIPProcessor, CLIPVisionModel
)
import os
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
from src.data_processing.chinese_text_augmentation import ChineseTextAugmenter

class FakeNewsDataPreprocessor:
    def __init__(self):
        """
        初始化预处理器
        """
        # 使用实际存在的数据集路径
        self.train_data_path = '/root/autodl-tmp/data/small_train.csv'
        self.test_data_path = '/root/autodl-tmp/data/small_test.csv'
        self.val_data_path = '/root/autodl-tmp/data/small_val.csv'
        self.images_dir = '/root/autodl-tmp/data/images'
        
        # 设置本地缓存目录
        cache_dir = '/root/autodl-tmp/model_cache_new'
        print(f'使用缓存目录: {cache_dir}')
        os.makedirs(cache_dir, exist_ok=True)
        
        # 加载tokenizer和processor
        print("正在加载模型...")
        def load_model(model_name, model_class):
            print(f'正在加载{model_name}...')
            try:
                # 使用tqdm显示模型加载进度
                with tqdm(desc=f'加载{model_name}', unit='B', unit_scale=True) as pbar:
                    # 首先检查模型文件大小
                    model_path = os.path.join(cache_dir, f'models--{model_name.replace("/", "--")}')
                    if os.path.exists(model_path):
                        total_size = sum(f.stat().st_size for f in Path(model_path).rglob('*') if f.is_file())
                        pbar.total = total_size
                    
                    # 加载模型
                    model = model_class.from_pretrained(
                        model_name,
                        cache_dir=cache_dir
                    )
                    
                    # 更新进度条到100%
                    if pbar.total is not None:
                        pbar.update(pbar.total - pbar.n)
                    
                print(f'{model_name} 加载成功')
                return model
            except Exception as e:
                print(f'{model_name} 加载失败: {e}')
                raise

        # 由于网络问题，直接使用简化的fallback模型
        print("网络不可用，使用简化的fallback模型...")
        
        # 创建简化的分词器
        class SimpleTokenizer:
            def __init__(self):
                self.vocab_size = 30522
                self.max_length = 512
                
            def __call__(self, text, max_length=512, padding=True, truncation=True, return_tensors='pt'):
                # 简单的字符级编码
                if isinstance(text, list):
                    encoded = []
                    for t in text:
                        tokens = [ord(c) % self.vocab_size for c in t[:max_length]]
                        if len(tokens) < max_length:
                            tokens.extend([0] * (max_length - len(tokens)))
                        encoded.append(tokens)
                    return {
                        'input_ids': torch.tensor(encoded),
                        'attention_mask': torch.ones(len(encoded), max_length)
                    }
                else:
                    tokens = [ord(c) % self.vocab_size for c in text[:max_length]]
                    if len(tokens) < max_length:
                        tokens.extend([0] * (max_length - len(tokens)))
                    return {
                        'input_ids': torch.tensor([tokens]),
                        'attention_mask': torch.ones(1, max_length)
                    }
        
        self.text_tokenizer = SimpleTokenizer()
        
        # 创建简化的文本模型
        class SimpleTextModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(30522, 768)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True),
                    num_layers=2
                )
                
            def forward(self, input_ids, attention_mask=None):
                x = self.embedding(input_ids)
                x = self.transformer(x)
                return type('obj', (object,), {'last_hidden_state': x})()
        
        self.text_model = SimpleTextModel()
        
        # 创建简化的图像处理器
        class SimpleImageProcessor:
            def __call__(self, images, return_tensors='pt'):
                if isinstance(images, list):
                    processed = []
                    for img in images:
                        if isinstance(img, str):
                            # 如果是路径，创建默认张量
                            processed.append(torch.randn(3, 224, 224))
                        else:
                            # 如果是PIL图像，转换为张量
                            import torchvision.transforms as transforms
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor()
                            ])
                            processed.append(transform(img))
                    return {'pixel_values': torch.stack(processed)}
                else:
                    if isinstance(images, str):
                        return {'pixel_values': torch.randn(1, 3, 224, 224)}
                    else:
                        import torchvision.transforms as transforms
                        transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor()
                        ])
                        return {'pixel_values': transform(images).unsqueeze(0)}
        
        self.clip_processor = SimpleImageProcessor()
        
        # 创建简化的图像模型
        class SimpleImageModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision_model = nn.Sequential(
                    nn.Conv2d(3, 64, 7, 2, 3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((7, 7)),
                    nn.Flatten(),
                    nn.Linear(64 * 7 * 7, 768)
                )
                
            def forward(self, pixel_values):
                x = self.vision_model(pixel_values)
                return type('obj', (object,), {'pooler_output': x})()
        
        self.image_model = SimpleImageModel()
        print("Fallback模型创建成功")
        
        # 模型模式将在训练时设置
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.text_model = self.text_model.to(self.device)
        self.image_model = self.image_model.to(self.device)
        
        print("正在加载数据集...")
        # 加载CSV格式的数据集
        try:
            self.train_df = pd.read_csv(self.train_data_path, encoding='utf-8')
            self.test_df = pd.read_csv(self.test_data_path, encoding='utf-8')
            self.val_df = pd.read_csv(self.val_data_path, encoding='utf-8')
            
            # 数据清洗
            def clean_dataframe(df):
                # 删除缺失值
                df = df.dropna(subset=['text', 'path', 'label'])
                # 确保text列是字符串类型
                df['text'] = df['text'].astype(str)
                # 过滤掉空字符串
                df = df[df['text'].str.strip() != '']
                return df.reset_index(drop=True)
            
            self.train_df = clean_dataframe(self.train_df)
            self.test_df = clean_dataframe(self.test_df)
            self.val_df = clean_dataframe(self.val_df)
            
            print(f"训练集加载完成，共有{len(self.train_df)}条记录")
            print(f"测试集加载完成，共有{len(self.test_df)}条记录")
            print(f"验证集加载完成，共有{len(self.val_df)}条记录")
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
        
    def get_text_features(self, text_input):
        """
        提取文本特征（CLS Token）
        """
        with torch.no_grad():
            outputs = self.text_model(**text_input)
        return outputs.last_hidden_state[:, 0, :]  # [batch_size, 768]
    
    def get_image_features(self, image_input):
        """
        提取图像特征（Pooled Output）
        """
        with torch.no_grad():
            outputs = self.image_model(**image_input)
        return outputs.pooler_output  # [batch_size, 512]
        
    def encode_text(self, text):
        """
        使用DistilBERT的tokenizer对文本进行编码
        """
        encoded = self.text_tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def encode_image(self, image_path):
        """
        使用CLIP的processor对图像进行编码
        """
        try:
            image = Image.open(image_path).convert('RGB')
            encoded = self.clip_processor(
                images=image,
                return_tensors="pt"
            )
            return {k: v.to(self.device) for k, v in encoded.items()}
        except Exception as e:
            print(f"处理图像时出错 {image_path}: {str(e)}")
            # 如果是简化模型，返回默认张量
            if hasattr(self.clip_processor, '__class__') and 'Simple' in self.clip_processor.__class__.__name__:
                return {'pixel_values': torch.randn(1, 3, 224, 224).to(self.device)}
            return None

class FakeNewsDataset(Dataset):
    def __init__(self, preprocessor, split='dev', max_samples=100, device='cuda', use_cache=True, augment_text=False):
        """
        初始化数据集
        Args:
            preprocessor: FakeNewsDataPreprocessor实例
            split: 数据集划分（'dev'或'test'）
            max_samples: 最大样本数量（用于调试）
            device: 设备（'cuda'或'cpu'）
            use_cache: 是否使用特征缓存
            augment_text: 是否使用文本增强
        """
        self.preprocessor = preprocessor
        self.split = split
        self.device = device
        self.missing_images = set()  # 用于记录缺失的图片
        self.use_cache = use_cache
        self.cache_dir = os.path.join(os.getcwd(), 'feature_cache')
        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # 文本增强设置
        self.augment_text = augment_text
        if augment_text and split == 'dev':  # 只在训练集上进行增强
            self.text_augmenter = ChineseTextAugmenter(
                synonym_prob=0.3,  # 30%的概率进行同义词替换
                synonym_percent=0.2  # 替换20%的词
            )
            print("已启用中文文本增强功能")
        
        # 根据划分选择数据集
        if split == 'train':
            self.df = preprocessor.train_df
        elif split == 'val':
            self.df = preprocessor.val_df
        else:
            self.df = preprocessor.test_df
        
        self.image_dir = preprocessor.images_dir
        
        # 将模型移动到指定设备
        self.preprocessor.text_model.to(device)
        self.preprocessor.image_model.to(device)
        
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
        
        # 统计数据集信息
        print(f"数据集大小: {len(self.df)}条记录")
        print("标签分布:")
        print(self.df['label'].value_counts())
        
        # 检查图片完整性
        total_images = 0
        missing_images = 0
        unique_image_ids = set()
        
        for _, row in self.df.iterrows():
            # 从path列获取图像路径
            image_path_field = row['path']
            if pd.notna(image_path_field):
                # 提取文件名作为image_id
                image_filename = os.path.basename(str(image_path_field))
                image_id = os.path.splitext(image_filename)[0]  # 去掉扩展名
                total_images += 1
                unique_image_ids.add(image_id)
                
                # 构建完整的图像路径
                full_image_path = os.path.join(self.image_dir, image_filename)
                if not os.path.exists(full_image_path):
                    missing_images += 1
                    self.missing_images.add(image_id)
        
        print(f"图片统计:")
        print(f"- 总图片数: {total_images}")
        print(f"- 唯一图片数: {len(unique_image_ids)}")
        print(f"- 缺失图片数: {missing_images}")
        print(f"- 缺失率: {missing_images/total_images*100:.2f}%")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 获取文本和标签
        text = row['text']
        label = int(row['label'])  # CSV中label已经是0或1
        
        # 如果是训练集且启用了文本增强，进行同义词替换
        if self.augment_text and self.split == 'train' and hasattr(self, 'text_augmenter') and random.random() < 0.5:  # 50%的概率使用增强文本
            text = self.text_augmenter.augment(text)
        
        # 处理文本
        text_encoding = self.preprocessor.encode_text(text)
        text_features = self.preprocessor.get_text_features(text_encoding)
        # 确保文本特征是2D张量 [batch_size=1, hidden_dim]
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)
        
        # 获取图像路径
        image_path_field = row['path']
        # 从路径中提取图像文件名
        if pd.notna(image_path_field):
            image_filename = os.path.basename(str(image_path_field))
            image_id = os.path.splitext(image_filename)[0]  # 去掉扩展名
            image_ids = [image_id]
        else:
            image_ids = []
        
        # 初始化图像特征
        image_features = None
        
        # 处理图像
        for image_id in image_ids:
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            try:
                if os.path.exists(image_path):
                    image_encoding = self.preprocessor.encode_image(image_path)
                    if image_encoding is not None:
                        image_encoding = {k: v.to(self.device) for k, v in image_encoding.items()}
                        features = self.preprocessor.get_image_features(image_encoding)
                        # 确保特征是2D张量 [batch_size=1, hidden_dim]
                        if len(features.shape) == 1:
                            features = features.unsqueeze(0)
                        image_features = features
                        break  # 找到一个有效图片就停止
                    else:
                        if image_id not in self.missing_images:
                            print(f'图片编码失败: {image_path}')
                            self.missing_images.add(image_id)
                else:
                    if image_id not in self.missing_images:
                        print(f'图片不存在: {image_path}')
                        self.missing_images.add(image_id)
            except Exception as e:
                if image_id not in self.missing_images:
                    print(f'处理图片时出错 {image_path}: {str(e)}')
                    self.missing_images.add(image_id)
        
        # 如果没有有效图片，使用零向量
        if image_features is None:
            image_features = torch.zeros((1, 768)).to(self.device)  # 使用与文本特征相同的维度
        
        return {
            'text_features': text_features,
            'image_features': image_features,
            'label': torch.tensor(label, device=self.device)
        }

class TransformerFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=64, num_labels=2, dropout=0.3):
        super().__init__()
        # 特征投影到同一维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # Transformer 编码层
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2,  # 拼接后的维度
            nhead=4,  # 注意力头数
            dim_feedforward=256,  # 前馈网络维度
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(dropout)
        # 分类器
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, text_features, image_features):
        # 投影特征
        text_proj = self.text_proj(text_features)  # [batch_size, hidden_dim]
        image_proj = self.image_proj(image_features)  # [batch_size, hidden_dim]
        
        # 拼接特征
        fused = torch.cat([text_proj, image_proj], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Transformer 融合
        fused = fused.unsqueeze(0)  # [1, batch_size, hidden_dim * 2]
        fused = self.transformer(fused)
        fused = fused.squeeze(0)  # [batch_size, hidden_dim * 2]
        
        # Dropout
        fused = self.dropout(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits

class MLPFusion(nn.Module):
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=256, num_labels=2):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(text_dim + image_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, text_features, image_features):
        fused = torch.cat([text_features, image_features], dim=1)  # [batch_size, 768+512]
        logits = self.fc(fused)
        return logits

def evaluate(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    val_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            text_features = batch['text_features'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].long().to(device)
            
            logits = model(text_features, image_features)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = val_loss / len(val_loader)
    
    return accuracy, f1, avg_loss

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda', weight_decay=1e-4):
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_model = None
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 训练循环
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            text_features = batch['text_features'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].long().to(device)
            
            optimizer.zero_grad()
            logits = model(text_features, image_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 验证
        val_acc, val_f1, val_loss = evaluate(model, val_loader, device)
        avg_train_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Average Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        print(f'Validation F1 Score: {val_f1:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict().copy()
    
    # 加载最佳模型
    model.load_state_dict(best_model)
    return model

def custom_collate_fn(batch):
    """
    自定义批处理函数，处理不同大小的张量
    """
    if len(batch) == 0:
        return {}
        
    # 检查特征维度
    text_dim = batch[0]['text_features'].shape[1]
    image_dim = batch[0]['image_features'].shape[1]
    
    # 确保所有批次中的特征维度一致
    for item in batch:
        assert item['text_features'].shape[1] == text_dim, f'文本特征维度不匹配: {item["text_features"].shape[1]} != {text_dim}'
        assert item['image_features'].shape[1] == image_dim, f'图像特征维度不匹配: {item["image_features"].shape[1]} != {image_dim}'
    
    text_features = torch.stack([item['text_features'].squeeze(0) for item in batch])
    image_features = torch.stack([item['image_features'].squeeze(0) for item in batch])
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    
    return {
        'text_features': text_features,
        'image_features': image_features,
        'label': labels
    }

def create_data_loaders(dataset, batch_size=16, train_ratio=0.8, num_workers=2):
    """
    创建训练集和验证集的数据加载器
    """
    # 计算分割点
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    
    # 分割数据集
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 暂时禁用多进程加载
        pin_memory=False,  # 数据已经在GPU上，禁用pin_memory
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=0,  # 暂时禁用多进程加载
        pin_memory=False,  # 数据已经在GPU上，禁用pin_memory
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader

def main():
    # 创建预处理器
    preprocessor = FakeNewsDataPreprocessor()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建小规模数据集用于测试
    dataset = FakeNewsDataset(preprocessor, split='train', max_samples=10, device=device, augment_text=True)  # 只使用10条数据进行测试
    
    # 创建数据加载器，使用小批量
    train_loader, val_loader = create_data_loaders(dataset, batch_size=2)
    
    # 创建融合模型（可以选择 TransformerFusion 或 MLPFusion）
    fusion_model = TransformerFusion()
    
    # 训练模型
    print('开始训练模型...')
    trained_model = train_model(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=2,  # 减少训练轮数用于测试
        device=device
    )
    
    # 在验证集上进行最终评估
    print('在验证集上进行最终评估...')
    val_acc, val_f1, val_loss = evaluate(trained_model, val_loader, device)
    print(f'最终验证集结果:')
    print(f'Accuracy: {val_acc:.4f}')
    print(f'F1 Score: {val_f1:.4f}')
    print(f'Loss: {val_loss:.4f}')
    
    # 保存模型
    torch.save(trained_model.state_dict(), 'fake_news_fusion_model.pth')
    print('模型已保存为 fake_news_fusion_model.pth')
    
    print(f"\n训练集大小: {len(train_loader.dataset)}条记录")
    print(f"验证集大小: {len(val_loader.dataset)}条记录")

if __name__ == "__main__":
    main()