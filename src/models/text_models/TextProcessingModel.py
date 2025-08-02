import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from transformers import DistilBertTokenizer, DistilBertModel, BertTokenizer, BertModel, CLIPTokenizer, CLIPTextModel
import platform
import warnings
import pandas as pd
from PIL import Image
import os
import time
import gc  # 导入垃圾回收模块以优化内存使用
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from pathlib import Path

# CUDA相关配置
CUDA_VISIBLE_DEVICES = "0"  # 使用的GPU编号，RTX 3090通常为0
CUDA_LAUNCH_BLOCKING = "1"  # 设置为1可以帮助调试CUDA错误
CUDA_BATCH_SIZE = 32  # CUDA环境下的默认批量大小
CUDA_NUM_WORKERS = 8  # CUDA环境下的数据加载器工作线程数

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['CUDA_LAUNCH_BLOCKING'] = CUDA_LAUNCH_BLOCKING

# 设置常量
CACHE_DIR = '/Users/wujianxiang/Documents/GitHub/models/model_cache'

class FakeNewsDataset(Dataset):
    def __init__(self, data_path, images_dir, bert_tokenizer, clip_tokenizer, 
                 max_length=512, clip_max_length=77, transform=None):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            images_dir: 图像目录路径
            bert_tokenizer: BERT分词器
            clip_tokenizer: CLIP分词器
            max_length: BERT最大序列长度
            clip_max_length: CLIP最大序列长度
            transform: 图像变换
        """
        self.data_path = data_path
        self.images_dir = images_dir
        self.bert_tokenizer = bert_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.max_length = max_length
        self.clip_max_length = clip_max_length
        self.transform = transform
        
        # 加载数据
        try:
            # 检测文件扩展名，决定分隔符
            if data_path.endswith('.csv'):
                # CSV格式，使用逗号分隔
                self.df = pd.read_csv(
                    data_path,
                    encoding='utf-8',
                    engine='python'  # 使用Python引擎解析，解决C解析器的错误
                )
                print(f"CSV数据集加载完成，共有{len(self.df)}条记录")
                print(f"列名: {list(self.df.columns)}")
            else:
                # 文本格式，使用制表符分隔
                self.df = pd.read_csv(
                    data_path,
                    delimiter='\t',
                    quoting=3,  # QUOTE_NONE
                    encoding='utf-8',
                    escapechar='\\',
                    engine='python'  # 使用Python引擎解析，解决C解析器的错误
                )
                print(f"文本数据集加载完成，共有{len(self.df)}条记录")
                print(f"列名: {list(self.df.columns)}")
            
            # 统一列名
            if 'image_id(s)' in self.df.columns:
                self.df = self.df.rename(columns={'image_id(s)': 'image_id'})
                
            # 确保必要的列存在
            required_columns = ['text', 'label']
            for col in required_columns:
                if col not in self.df.columns:
                    raise ValueError(f"数据集缺少必要的列: {col}")
            
            # 处理图像路径列
            if 'path' in self.df.columns:
                # 新格式，使用path列
                self.image_col = 'path'
            elif 'image_id' in self.df.columns:
                # 旧格式，使用image_id列
                self.image_col = 'image_id'
            elif 'images' in self.df.columns:
                # 可能的替代列名
                self.image_col = 'images'
            elif 'img' in self.df.columns:
                # 可能的替代列名
                self.image_col = 'img'
            elif 'photo' in self.df.columns:
                # 可能的替代列名
                self.image_col = 'photo'
            else:
                print(f"警告: 未找到图像路径列，可用的列: {list(self.df.columns)}")
                self.image_col = None
            
            # 将标签转换为数字
            if 'label' in self.df.columns:
                # 检查标签类型
                if self.df['label'].dtype == 'object':
                    # 如果是字符串类型，转换为数字
                    self.df['label_num'] = self.df['label'].apply(
                        lambda x: 1 if x.lower() in ['fake', 'false', '1', 'y', 'yes', 'true'] else 0
                    )
                    print("将文本标签转换为数字标签")
                else:
                    # 如果已经是数字类型，直接使用
                    self.df['label_num'] = self.df['label']
                    print("标签已经是数字类型")
            else:
                raise ValueError("数据集缺少标签列")
            
            # 打印标签分布
            label_counts = self.df['label_num'].value_counts()
            print(f"标签分布: {label_counts.to_dict()}")
            
        except Exception as e:
            print(f"加载数据集时出错: {str(e)}")
            raise
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['text'])
        label = int(row['label_num'])
        
        # 使用BERT分词器处理文本
        bert_encoding = self.bert_tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 使用CLIP分词器处理文本
        clip_encoding = self.clip_tokenizer(
            text,
            max_length=self.clip_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理图像路径
        image_path = None
        if self.image_col is not None and self.image_col in row:
            # 可能有多个图像路径，用逗号分隔
            image_paths = str(row[self.image_col]).split(',')
            if len(image_paths) > 0:
                image_path = image_paths[0].strip()  # 暂时只使用第一张图像
        
        return {
            'bert_input_ids': bert_encoding['input_ids'].squeeze(0),
            'bert_attention_mask': bert_encoding['attention_mask'].squeeze(0),
            'clip_input_ids': clip_encoding['input_ids'].squeeze(0),
            'clip_attention_mask': clip_encoding['attention_mask'].squeeze(0),
            'image_path': image_path,
            'label': torch.tensor(label)
        }

class TextProcessingModel(nn.Module):
    def __init__(self, num_classes=2, bert_model_path='distilbert-base-uncased', 
                 clip_model_path='openai/clip-vit-base-patch32', use_local_models=False):
        """
        初始化文本处理模型
        
        Args:
            num_classes: 分类类别数
            bert_model_path: BERT模型路径
            clip_model_path: CLIP模型路径
            use_local_models: 是否使用本地模型
        """
        super(TextProcessingModel, self).__init__()
        
        # 加载预训练模型，支持本地模型加载
        local_files_only = use_local_models
        
        # 初始化维度
        self.bert_dim = 768
        self.clip_dim = 512  # CLIP文本编码器输出维度
        
        # 尝试使用预训练模型
        try:
            # 检查是否使用中文BERT模型
            is_chinese_bert = 'chinese' in bert_model_path.lower()
            
            # 使用BERT模型提取深层语义信息
            print(f'尝试加载BERT模型: {bert_model_path}, local_files_only={local_files_only}, is_chinese_bert={is_chinese_bert}')
            if is_chinese_bert:
                self.bert_encoder = BertModel.from_pretrained(bert_model_path, local_files_only=local_files_only)
                print('成功加载中文BERT模型')
            else:
                self.bert_encoder = DistilBertModel.from_pretrained(bert_model_path, local_files_only=local_files_only)
                print('成功加载DistilBERT模型')
            
            # 使用CLIP文本编码器提取文本的视觉关联特征
            print(f'尝试加载CLIP模型: {clip_model_path}, local_files_only={local_files_only}')
            self.clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_path, local_files_only=local_files_only)
            print('成功加载CLIP模型')
        except Exception as e:
            print(f'加载预训练模型失败: {e}')
            print('创建新的模型架构...')
            
            # 如果加载失败，创建新的模型架构
            # 这里我们创建一个简化的模型，而不是使用预训练模型
            from torch.nn import TransformerEncoderLayer, TransformerEncoder
            
            # 创建简化的BERT模型
            vocab_size = 21128 if 'chinese' in bert_model_path.lower() else 30522  # 中文BERT词汇表大小或英文DistilBERT词汇表大小
            self.bert_encoder = nn.Sequential(
                nn.Embedding(vocab_size, self.bert_dim),
                TransformerEncoder(TransformerEncoderLayer(
                    d_model=self.bert_dim, nhead=8, dim_feedforward=3072, batch_first=True), num_layers=2)
            )
            
            # 创建简化的CLIP文本模型
            self.clip_text_encoder = nn.Sequential(
                nn.Embedding(49408, self.clip_dim),  # CLIP词汇表大小
                TransformerEncoder(TransformerEncoderLayer(
                    d_model=self.clip_dim, nhead=8, dim_feedforward=2048, batch_first=True), num_layers=2)
            )
            
            print('创建了简化的模型架构')
        
        # 特征对齐层 - 将BERT和CLIP特征映射到相同的维度空间
        self.bert_alignment = nn.Linear(self.bert_dim, 512)
        self.clip_alignment = nn.Linear(self.clip_dim, 512)  # 修改CLIP维度
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 512),  # 融合对齐后的特征
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, bert_input_ids, bert_attention_mask, clip_input_ids, clip_attention_mask):
        """
        前向传播函数
        
        Args:
            bert_input_ids: BERT输入ID [batch_size, seq_len]
            bert_attention_mask: BERT注意力掩码 [batch_size, seq_len]
            clip_input_ids: CLIP输入ID [batch_size, seq_len]
            clip_attention_mask: CLIP注意力掩码 [batch_size, seq_len]
            
        Returns:
            outputs: 模型输出
            bert_features: BERT特征 [batch_size, bert_dim]
            clip_features: CLIP特征 [batch_size, clip_dim]
        """
        try:
            # 对于预训练模型和简化模型使用不同的前向传播逻辑
            if isinstance(self.bert_encoder, nn.Sequential):
                # 简化模型的前向传播
                bert_embeds = self.bert_encoder[0](bert_input_ids)
                bert_outputs = self.bert_encoder[1](bert_embeds, src_key_padding_mask=~bert_attention_mask.bool())
                bert_features = bert_outputs.mean(dim=1)  # 平均池化得到句子表示
                
                clip_embeds = self.clip_text_encoder[0](clip_input_ids)
                clip_outputs = self.clip_text_encoder[1](clip_embeds, src_key_padding_mask=~clip_attention_mask.bool())
                clip_features = clip_outputs.mean(dim=1)  # 平均池化得到句子表示
            else:
                # 预训练模型的前向传播
                bert_outputs = self.bert_encoder(input_ids=bert_input_ids, attention_mask=bert_attention_mask)
                bert_features = bert_outputs.last_hidden_state.mean(dim=1)  # 平均池化得到句子表示
                
                clip_outputs = self.clip_text_encoder(input_ids=clip_input_ids, attention_mask=clip_attention_mask)
                clip_features = clip_outputs.last_hidden_state.mean(dim=1)  # 平均池化得到句子表示
            
            # 特征拼接
            combined_features = torch.cat([self.bert_alignment(bert_features), self.clip_alignment(clip_features)], dim=1)
            outputs = self.classifier(self.fusion(combined_features))
            
            return outputs, bert_features, clip_features
        except Exception as e:
            print(f'预训练模型前向传播失败: {e}')
            # 创建空特征以避免训练崩溃
            batch_size = bert_input_ids.size(0)
            bert_features = torch.zeros(batch_size, self.bert_dim).to(bert_input_ids.device)
            clip_features = torch.zeros(batch_size, self.clip_dim).to(bert_input_ids.device)
            outputs = torch.zeros(batch_size, 2).to(bert_input_ids.device)  # 假设二分类
            
            return outputs, bert_features, clip_features

def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=2e-5, weight_decay=1e-4, gradient_accumulation_steps=1, save_interval=1):
    # 在训练开始前触发垃圾回收
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 设置是否使用大数据集模式
    large_dataset = len(train_loader.dataset) > 100
    
    # 如果是大数据集，每个轮次中定期触发垃圾回收
    gc_interval = 50 if large_dataset else 0  # 每50个批次触发一次垃圾回收
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    # 将模型移动到设备
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    
    # 对于CUDA，我们可以使用更高效的设置
    if device.type == 'cuda':
        # 在CUDA上，使用默认参数通常效果很好
        print("使用CUDA优化设置")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        # 在CPU上，可能需要调整一些参数
        print("使用CPU优化设置")
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # 最佳模型保存
    best_val_f1 = 0.0
    best_model_path = 'best_text_processing_model.pth'
    
    # 性能监控变量
    batch_times = []
    import time
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        train_loss = 0.0
        
        # 记录每个epoch的开始时间
        epoch_start_time = time.time()
        
        # 进度条
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, batch in enumerate(train_pbar):
            # 对于大数据集，定期触发垃圾回收
            if gc_interval > 0 and batch_idx % gc_interval == 0 and batch_idx > 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"\n批次 {batch_idx}/{len(train_loader)} 触发垃圾回收")
            
            # 获取数据
            try:
                bert_input_ids = batch['bert_input_ids'].to(device)
                bert_attention_mask = batch['bert_attention_mask'].to(device)
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                labels = batch['label'].to(device)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"CUDA内存不足，跳过此批次: {e}")
                    # 尝试清理内存
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            
            # 小数据集时，打印更多调试信息
            if batch_idx % 5 == 0 and len(train_loader) < 20:
                print(f"\n批次 {batch_idx+1}/{len(train_loader)}")
                print(f"BERT输入形状: {bert_input_ids.shape}")
                print(f"CLIP输入形状: {clip_input_ids.shape}")
                print(f"标签: {labels}")
            
            # 前向传播
            logits, _, _ = model(
                bert_input_ids, bert_attention_mask, 
                clip_input_ids, clip_attention_mask
            )
            
            # 计算损失
            loss = criterion(logits, labels)
            
            # 反向传播和优化 (支持梯度累积)
            loss = loss / gradient_accumulation_steps  # 如果使用梯度累积，则缩放损失
            loss.backward()
            
            # 记录批次处理时间
            batch_start_time = time.time()
            
            # 每gradient_accumulation_steps步进行一次优化
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # 更新训练损失
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        # 进度条
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
        
        with torch.no_grad():
            for batch in val_pbar:
                # 获取数据
                bert_input_ids = batch['bert_input_ids'].to(device)
                bert_attention_mask = batch['bert_attention_mask'].to(device)
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                logits, _, _ = model(
                    bert_input_ids, bert_attention_mask, 
                    clip_input_ids, clip_attention_mask
                )
                
                # 计算损失
                loss = criterion(logits, labels)
                
                # 更新验证损失
                val_loss += loss.item()
                
                # 获取预测结果
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # 计算平均验证损失
        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)
        
        # 计算验证指标
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 计算epoch总时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        print(f'  Epoch Time: {epoch_time:.2f}s, Avg Batch Time: {np.mean(batch_times):.4f}s')
        
        # 每个轮次结束后触发垃圾回收以减少内存使用
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_f1': val_f1,
            }, best_model_path)
            print(f'  保存最佳模型，F1值: {val_f1:.4f}')
            
        # 定期保存模型
        if (epoch + 1) % save_interval == 0:
            model_save_path = f'/Users/wujianxiang/Documents/GitHub/models/model_cache/fake_news_model_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), model_save_path)
            print(f'  模型已保存到 {model_save_path}')
    
    # 完全跳过加载最佳模型的步骤，直接使用当前模型
    # 这样可以避免超时问题，因为我们已经在训练过程中保存了每个轮次的模型
    print(f"跳过加载最佳模型的步骤，直接使用当前模型")
    print(f"最终训练轮次: {epoch+1}, 最终验证F1: {val_f1:.4f}")
    
    # 如果在未来需要加载最佳模型，可以使用以下代码：
    # try:
    #     import signal
    #     class TimeoutException(Exception): pass
    #     
    #     def timeout_handler(signum, frame):
    #         raise TimeoutException("Loading model timed out")
    #     
    #     # 设置10秒超时
    #     signal.signal(signal.SIGALRM, timeout_handler)
    #     signal.alarm(10)
    #     
    #     print(f"尝试加载最佳模型: {best_model_path}")
    #     checkpoint = torch.load(best_model_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print(f"成功加载最佳模型，验证F1: {checkpoint['val_f1']:.4f}")
    #     
    #     # 取消超时
    #     signal.alarm(0)
    # except (TimeoutException, TimeoutError, FileNotFoundError, RuntimeError, KeyError) as e:
    #     print(f"加载最佳模型失败: {e}")
    #     print("继续使用当前模型...")
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        metrics: 评估指标
    """
    # 将模型移动到设备
    model = model.to(device)
    
    # 评估模式
    model.eval()
    all_preds = []
    all_labels = []
    
    # 进度条
    test_pbar = tqdm(test_loader, desc='Evaluating')
    
    with torch.no_grad():
        for batch in test_pbar:
            # 获取数据
            try:
                bert_input_ids = batch['bert_input_ids'].to(device)
                bert_attention_mask = batch['bert_attention_mask'].to(device)
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                labels = batch['label'].to(device)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"CUDA内存不足，跳过此批次: {e}")
                    # 尝试清理内存
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            
            # 前向传播
            logits, _, _ = model(
                bert_input_ids, bert_attention_mask, 
                clip_input_ids, clip_attention_mask
            )
            
            # 获取预测结果
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算评估指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1': f1_score(all_labels, all_preds, average='weighted')
    }
    
    # 打印评估指标
    print('评估指标:')
    for metric_name, metric_value in metrics.items():
        print(f'  {metric_name}: {metric_value:.4f}')
    
    # 绘制混淆矩阵 (使用英文标签)
    cm = confusion_matrix(all_labels, all_preds)
    
    # 在绘图前触发垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 优化内存使用 - 使用较小的图像尺寸
    plt.figure(figsize=(6, 4), dpi=100)
    
    # 使用英文标签
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 优化内存使用 - 使用较低的DPI保存
    plt.savefig('confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close('all')  # 关闭所有图形以释放内存
    
    return metrics

def plot_training_history(history):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
    """
    # 创建图形 - 减小尺寸以节省内存
    plt.figure(figsize=(10, 3), dpi=100)
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # 绘制指标
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.plot(history['val_f1'], label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    
    # 保存图形 - 使用优化参数减少内存使用
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=100, bbox_inches='tight')
    plt.close('all')  # 关闭所有图形以释放内存

def setup_cuda_device():
    """
    设置CUDA设备并处理可能的兼容性问题，优先使用CUDA（适用于RTX 3090）
    
    Returns:
        device: 可用的设备（cuda或cpu）
    """
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用CUDA设备加速: {torch.cuda.get_device_name(0)}")
        # 打印CUDA设备信息
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备属性: {torch.cuda.get_device_properties(0)}")
        
        # 设置CUDA性能参数
        torch.backends.cudnn.benchmark = True  # 启用cudnn自动调优
        torch.backends.cudnn.deterministic = False  # 关闭确定性模式以提高性能
        
        return device
    
    # CUDA不可用的情况
    print("CUDA不可用，请检查GPU驱动和CUDA安装")
    
    # 使用CPU
    device = torch.device('cpu')
    print("使用CPU设备")
    return device

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查MPS是否可用
    print(f"PyTorch版本: {torch.__version__}")
    print(f"系统平台: {platform.platform()}")
    print(f"MPS是否可用: {torch.backends.mps.is_available()}")
    print(f"MPS是否已构建: {torch.backends.mps.is_built()}")
    
    # 设置设备 - 支持MPS加速
    device = setup_cuda_device()
    print(f"使用设备: {device}")
    
    # 在开始前先清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 设置数据路径
    data_dir = '/Users/wujianxiang/Documents/GitHub/models/Data/twitter_dataset'
    train_data_path = os.path.join(data_dir, 'devset', 'posts.txt')
    train_images_dir = os.path.join(data_dir, 'devset', 'images')
    test_data_path = os.path.join(data_dir, 'testset', 'posts_groundtruth.txt')
    test_images_dir = os.path.join(data_dir, 'testset', 'images')
    
    # 加载分词器 - 使用本地缓存的模型
    print('加载分词器...')
    
    # 根据实际的文件结构设置正确的模型路径
    use_offline_mode = True
    
    # 使用预训练好的模型
    bert_model_path = 'distilbert-base-uncased'
    clip_model_path = 'openai/clip-vit-base-patch32'
    
    # 如果需要使用本地模型，可以设置正确的路径
    clip_local_path = '/Users/wujianxiang/Documents/GitHub/models/model_cache/clip/openai_clip-vit-base-patch32_processor'
    
    print(f'尝试加载模型...')
    
    try:
        # 尝试使用默认模型
        bert_tokenizer = DistilBertTokenizer.from_pretrained(bert_model_path, local_files_only=use_offline_mode)
        clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_path, local_files_only=use_offline_mode)
        print('成功加载分词器')
    except Exception as e:
        print(f'加载分词器失败: {e}')
        print('尝试使用已下载的模型...')
        
        # 尝试使用已下载的模型
        try:
            clip_tokenizer = CLIPTokenizer.from_pretrained(clip_local_path)
            bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
            print('成功加载本地CLIP分词器')
        except Exception as e:
            print(f'加载本地CLIP分词器失败: {e}')
            # 如果仍然失败，尝试使用在线模式
            print('尝试使用在线模式...')
            bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=False)
            clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', local_files_only=False)
    
    # 创建数据集
    print('创建数据集...')
    train_dataset = FakeNewsDataset(
        train_data_path, train_images_dir,
        bert_tokenizer, clip_tokenizer
    )
    test_dataset = FakeNewsDataset(
        test_data_path, test_images_dir,
        bert_tokenizer, clip_tokenizer
    )
    
    # 是否使用小型测试数据集
    use_small_dataset = False  # 使用全部数据集
    
    if use_small_dataset:
        # 只使用少量数据进行测试 - 进一步减少数据量以减少内存使用
        small_size = 30  # 减少到只使用30条数据进行测试
        indices = list(range(min(small_size, len(train_dataset))))
        train_dataset = Subset(train_dataset, indices)
        
        test_indices = list(range(min(small_size // 3, len(test_dataset))))
        test_dataset = Subset(test_dataset, test_indices)
        
        # 手动触发垃圾回收
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 划分训练集和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        # 使用完整数据集
        # 划分训练集和验证集
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f'训练集大小: {len(train_dataset)}')
    print(f'验证集大小: {len(val_dataset)}')
    print(f'测试集大小: {len(test_dataset)}')
    
    # 设置批处理大小和数据加载器参数
    if use_small_dataset:
        batch_size = 16 if device.type == 'cuda' else 8  # 对于小数据集，使用较大的批次大小
    else:
        batch_size = CUDA_BATCH_SIZE if device.type == 'cuda' else 8  # 对于全部数据集，使用更大的批次大小
    
    # 数据加载器设置
    num_workers = CUDA_NUM_WORKERS if device.type == 'cuda' else 2
    pin_memory = True if device.type == 'cuda' else False
    
    print(f"批处理大小: {batch_size}, 工作线程数: {num_workers}, 内存固定: {pin_memory}")
    
    # 创建数据加载器 - 优化内存使用
    # 对于MPS，我们需要减少num_workers并禁用预取以减少内存使用
    prefetch_factor = None if num_workers == 0 else 2  # 如果num_workers为0，则不使用prefetch
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory,  # 禁用pin_memory以减少内存使用
        persistent_workers=False  # 禁用持久化worker以减少内存使用
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,  # 禁用pin_memory以减少内存使用
        persistent_workers=False  # 禁用持久化worker以减少内存使用
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory,  # 禁用pin_memory以减少内存使用
        persistent_workers=False  # 禁用持久化worker以减少内存使用
    )
    
    # 创建模型
    print('创建模型...')
    
    # 加载预训练模型权重
    pretrained_model_path = '/Users/wujianxiang/Documents/GitHub/models/model_cache/fine_tuning_DistilBERT.pth'
    # 测试时跳过加载预训练模型
    load_pretrained = False
    
    if os.path.exists(pretrained_model_path):
        print(f'发现预训练模型: {pretrained_model_path}，但测试时跳过加载')
    
    # 创建模型
    model = TextProcessingModel(num_classes=2, 
                             bert_model_path=bert_model_path,
                             clip_model_path=clip_model_path,
                             use_local_models=use_offline_mode)
    
    # 如果有预训练模型，尝试加载
    if load_pretrained:
        try:
            model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
            print('成功加载预训练模型权重')
        except Exception as e:
            print(f'加载预训练模型失败: {e}')
    
    # 设置训练参数
    print("设置训练参数...")
    
    # 设置梯度累积步数 - 在CUDA上可以使用较小的值
    gradient_accumulation_steps = 1 if device.type == 'cuda' else 4
    
    # 根据设备类型优化训练参数
    if device.type == 'cuda':
        print('在CUDA上进行训练优化')
        # 对于CUDA，可以使用更大的学习率和更少的梯度累积步数
        lr = 2e-5  # 标准学习率在CUDA上通常效果很好
        weight_decay = 1e-4  # 标准权重衰减
        save_interval = 1  # 每个epoch保存一次模型
    else:
        print('在CPU上进行训练优化')
        # 对于CPU，使用较小的学习率和更多的梯度累积步数
        lr = 1e-5  # 较小的学习率
        weight_decay = 5e-5  # 较大的权重衰减
        save_interval = 2  # 每两个epoch保存一次模型
        gradient_accumulation_steps = 8  # 使用更大的梯度累积步数
    
    # 使用较少的训练轮数
    num_epochs = 1 if use_small_dataset else 5  # 对于全部数据集，使用5轮训练以平衡效果和内存使用
    print(f'设置训练轮数: {num_epochs}')
    
    # 手动触发垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # 对于全部数据集，增加梯度累积步数以减少内存使用
    if not use_small_dataset:
        gradient_accumulation_steps = 8 if device.type == 'mps' else 4  # 使用更大的梯度累积步数
        print(f'对全部数据集使用更大的梯度累积步数: {gradient_accumulation_steps}')
    
    model, history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=num_epochs, lr=lr, weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_interval=save_interval
    )
    
    # 添加错误处理以确保程序能够继续运行
    try:
        # 绘制训练历史
        print('绘制训练历史...')
        plot_training_history(history)
        print('训练历史绘制完成')
        
        # 手动触发垃圾回收
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 评估模型
        print('评估模型...')
        metrics = evaluate_model(model, test_loader, device)
        print('模型评估完成')
    except Exception as e:
        print(f'出现错误: {e}')
        import traceback
        traceback.print_exc()
    
    print('训练完成!')

if __name__ == '__main__':
    main()
