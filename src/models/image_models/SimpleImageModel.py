"""
简化版图像处理模型，专注于使用极小数据集进行测试
"""

import os
import gc
import time
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import CLIPProcessor, CLIPVisionModel
import seaborn as sns
from tqdm import tqdm

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

class SimpleFakeNewsDataset(Dataset):
    """
    简化版假新闻数据集，只加载少量数据
    """
    def __init__(self, data_path, images_dir, max_samples=20):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            images_dir: 图像目录路径
            max_samples: 最大样本数
        """
        self.images_dir = images_dir
        self.samples = []
        
        # 加载数据
        try:
            df = pd.read_csv(
                data_path,
                delimiter='\t',
                quoting=3,
                encoding='utf-8',
                escapechar='\\',
                engine='python'
            )
            print(f"成功加载数据: {len(df)} 条记录")
            
            # 处理每条记录
            count = 0
            for idx, row in df.iterrows():
                if count >= max_samples:
                    break
                    
                try:
                    # 检查图像ID列
                    if 'image_id(s)' in row:
                        image_id_col = 'image_id(s)'
                    elif 'image_id' in row:
                        image_id_col = 'image_id'
                    else:
                        continue
                        
                    # 获取图像ID
                    image_ids = str(row[image_id_col]).split(',')
                    image_path = None
                    
                    # 查找有效图像
                    for img_id in image_ids:
                        img_path = os.path.join(self.images_dir, f"{img_id.strip()}.jpg")
                        if os.path.exists(img_path):
                            image_path = img_path
                            break
                    
                    if image_path:
                        # 获取标签
                        label = 1 if row['label'] == 'fake' else 0
                        self.samples.append((image_path, label))
                        count += 1
                        
                except Exception as e:
                    pass
                    
            print(f"加载了 {len(self.samples)} 个有效样本")
            
        except Exception as e:
            print(f"加载数据失败: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        
        try:
            # 加载并预处理图像
            image = Image.open(image_path).convert('RGB')
            # 调整大小以减少内存使用
            image = image.resize((224, 224))
            
            # 转换为张量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image_tensor = transform(image)
            
            return image_tensor, torch.tensor(label, dtype=torch.long)
            
        except Exception as e:
            print(f"加载图像 {image_path} 时出错: {e}")
            # 返回空白图像
            return torch.zeros((3, 224, 224)), torch.tensor(label, dtype=torch.long)

class SimpleImageModel(nn.Module):
    """
    简化版图像处理模型
    """
    def __init__(self, num_classes=2):
        super(SimpleImageModel, self).__init__()
        
        # 加载预训练的ResNet-50
        print("加载ResNet-50模型...")
        resnet = models.resnet50(pretrained=True)
        # 移除最后的全连接层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        print("成功加载ResNet-50模型")
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)

def train_model(model, train_loader, val_loader, device, num_epochs=2):
    """
    训练模型
    """
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        print(f"Epoch {epoch+1}/{num_epochs} [Train]:")
        
        # 训练一个epoch
        for i, (images, labels) in enumerate(train_loader):
            # 将数据移动到设备
            images = images.to(device)
            labels = labels.to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 累加损失
            train_loss += loss.item()
            
            # 每处理完一个批次，打印进度
            print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # 清理内存
            if (i + 1) % 2 == 0:
                gc.collect()
                if device.type == 'mps':
                    torch.mps.empty_cache()
        
        # 计算平均训练损失
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        
        print(f"Epoch {epoch+1}/{num_epochs} [Val]:")
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                # 获取预测结果
                _, preds = torch.max(outputs, 1)
                
                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算验证指标
        val_loss /= len(val_loader)
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        
        # 保存历史
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        
        # 打印epoch结果
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Val F1: {val_f1:.4f}")
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    评估模型
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def setup_device():
    """
    设置设备
    """
    if torch.cuda.is_available():
        print("使用CUDA设备加速")
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("使用MPS设备加速")
        return torch.device('mps')
    else:
        print("使用CPU")
        return torch.device('cpu')

def main():
    """
    主函数
    """
    # 设置设备
    device = setup_device()
    print(f"使用设备: {device}")
    
    # 设置数据路径
    data_dir = '/Users/wujianxiang/Documents/GitHub/models/Data/twitter_dataset'
    train_data_path = os.path.join(data_dir, 'devset/posts.txt')
    train_images_dir = os.path.join(data_dir, 'devset/images')
    test_data_path = os.path.join(data_dir, 'testset/posts_groundtruth.txt')
    test_images_dir = os.path.join(data_dir, 'testset/images')
    
    # 创建数据集 - 使用极小数据集
    print("创建训练数据集...")
    train_dataset = SimpleFakeNewsDataset(train_data_path, train_images_dir, max_samples=20)
    
    # 分割训练集和验证集
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建测试数据集
    print("创建测试数据集...")
    test_dataset = SimpleFakeNewsDataset(test_data_path, test_images_dir, max_samples=10)
    print(f"测试集大小: {len(test_dataset)}")
    
    # 创建数据加载器
    batch_size = 2  # 使用非常小的批量大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print("创建模型...")
    model = SimpleImageModel(num_classes=2)
    model = model.to(device)
    print("模型创建完成")
    
    # 训练模型
    print("开始训练模型...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=2
    )
    print("模型训练完成")
    
    # 评估模型
    print("评估模型...")
    if len(test_dataset) > 0:
        metrics = evaluate_model(model, test_loader, device)
        
        # 打印最终评估结果
        print("\n最终评估结果:")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f}")
        print(f"F1值: {metrics['f1']:.4f}")
    else:
        print("测试集为空，跳过评估步骤")
    
    print("模型测试完成")
    
    # 保存模型
    print("保存模型...")
    model_dir = '/Users/wujianxiang/Documents/GitHub/models/model_cache'
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'simple_fake_news_model.pth'))
    print("模型保存完成")

if __name__ == '__main__':
    main()
