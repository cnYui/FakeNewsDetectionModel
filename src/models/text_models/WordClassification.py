import torch
import torch.nn as nn
from transformers import DistilBertModel, CLIPVisionModel
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
from tqdm import tqdm
import os

class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 打印一个数据样例
        print('数据样例:')
        print('文本:', str(texts[0]))
        print('标签:', int(labels[0]))

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # 处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label)
        }

class TextOnlyFakeNewsDetector(nn.Module):
    def __init__(self, num_classes=2):
        super(TextOnlyFakeNewsDetector, self).__init__()
        
        # 使用更小的BERT模型
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_dim = 768
        
        # 简化分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.text_dim, 256),  # 减少隐藏层大小
            nn.ReLU(),
            nn.Dropout(0.3),  # 增加Dropout以减少过拟合
            nn.Linear(256, num_classes)
        )
    
    def forward(self, input_ids, attention_mask):
        # 文本特征提取
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用[CLS]标记的输出作为文本表示
        text_features = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        
        # 分类
        logits = self.classifier(text_features)
        return logits

def train_text_model(model, train_loader, val_loader, num_epochs=10, device='cuda', save_dir='model_checkpoints'):
    print('\n开始训练...')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)  # 调整学习率和权重衰减
    
    # 创建模型保存目录
    os.makedirs(save_dir, exist_ok=True)
    model_save_path = os.path.join(save_dir, 'best_model.pth')
    
    best_accuracy = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # 添加进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 更新进度条后缀
            progress_bar.set_postfix({'loss': loss.item()})
        
        train_accuracy = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_accuracy = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')
        
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f'保存最佳模型到 {model_save_path}，验证准确率: {best_accuracy:.2f}%')
        
        print('--------------------')

def main():
    # 设置设备（支持CUDA、MPS和CPU）
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'使用设备: {device}')
    
    # 加载CSV数据集
    print('正在加载数据集...')
    df = pd.read_csv('Data/FakeNewsNet.csv')
    
    # 随机采样部分数据进行训练
    sample_size = min(5000, len(df))  # 最多使用5000条数据
    df = df.sample(n=sample_size, random_state=42)
    
    print('数据集加载完成')
    print(f'采样后数据集大小: {len(df)} 条记录')
    
    # 显示数据集的基本信息
    print('\n数据集信息：')
    print(df.info())
    print('\n标签分布：')
    print(df['real'].value_counts())
    
    # 准备数据
    texts = df['title'].values
    labels = df['real'].values
    
    # 初始化分词器
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    # 使用交叉验证
    use_cross_validation = True
    
    if use_cross_validation:
        # 交叉验证
        print('\n使用交叉验证进行训练和评估...')
        n_splits = 5
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_val_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(texts)):
            print(f'\n开始训练第 {fold+1}/{n_splits} 折...')
            
            # 分割数据
            train_texts = texts[train_idx]
            train_labels = labels[train_idx]
            val_texts = texts[val_idx]
            val_labels = labels[val_idx]
            
            # 创建数据集
            train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
            val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer)
            
            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=8)
            
            # 初始化模型
            model = TextOnlyFakeNewsDetector()
            
            # 训练模型
            model_save_dir = f'model_checkpoints/fold_{fold+1}'
            train_text_model(model, train_loader, val_loader, num_epochs=10, device=device, save_dir=model_save_dir)
            
            # 评估模型
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            fold_val_scores.append(val_accuracy)
            print(f'第 {fold+1} 折验证准确率: {val_accuracy:.2f}%')
        
        print('\n交叉验证结果:')
        for i, score in enumerate(fold_val_scores):
            print(f'第 {i+1} 折: {score:.2f}%')
        print(f'平均验证准确率: {sum(fold_val_scores)/len(fold_val_scores):.2f}%')
    
    else:
        # 标准训练流程
        # 分割训练集和验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, 
            labels,
            test_size=0.2,
            random_state=42
        )
        
        # 创建数据集
        train_dataset = FakeNewsDataset(train_texts, train_labels, tokenizer)
        val_dataset = FakeNewsDataset(val_texts, val_labels, tokenizer)
        
        # 创建数据加载器
        # 使用单进程数据加载
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8)
        
        # 初始化模型
        model = TextOnlyFakeNewsDetector()
        
        # 训练模型
        print('\n开始训练...')
        train_text_model(model, train_loader, val_loader, num_epochs=10, device=device, save_dir='model_checkpoints/standard')

if __name__ == '__main__':
    main()