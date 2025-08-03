#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整数据集训练脚本
使用完整的训练数据集重新训练多模态假新闻检测模型
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from tqdm import tqdm
import pandas as pd

# 添加src目录到Python路径
sys.path.append('/root/models/src')

# 直接在文件中定义模型类
class MultiModalFusionModel(nn.Module):
    """
    多模态融合模型
    """
    def __init__(self, text_dim=768, image_dim=768, hidden_dim=512, num_classes=2, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.image_proj = nn.Linear(image_dim, hidden_dim)
        
        # 多层融合网络
        self.fusion_layers = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.classifier = nn.Linear(hidden_dim // 2, num_classes)
        
    def forward(self, text_features, image_features):
        # 投影到相同维度
        text_proj = self.text_proj(text_features.squeeze(1))
        image_proj = self.image_proj(image_features.squeeze(1))
        
        # 融合特征
        fused = torch.cat([text_proj, image_proj], dim=1)
        fused = self.fusion_layers(fused)
        
        # 分类
        logits = self.classifier(fused)
        return logits
from src.models.MultiModalFakeNewsDetector import FakeNewsDataset, FakeNewsDataPreprocessor

def clean_dataframe(df):
    """
    清洗数据框，删除包含NaN值的行并处理数据类型
    """
    print(f"清洗前数据集大小: {len(df)}")
    
    # 删除关键列中包含NaN值的行
    df = df.dropna(subset=['text', 'path', 'label'])
    
    # 将text列转换为字符串类型
    df['text'] = df['text'].astype(str)
    
    # 过滤掉空字符串
    df = df[df['text'].str.strip() != '']
    
    print(f"清洗后数据集大小: {len(df)}")
    return df

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        text_features = batch_data['text_features'].to(device)
        image_features = batch_data['image_features'].to(device)
        labels = batch_data['label'].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(text_features, image_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算预测结果
        _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # 计算训练指标
    train_accuracy = accuracy_score(all_labels, all_predictions)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, train_accuracy

def validate_epoch(model, dataloader, criterion, device):
    """
    验证一个epoch
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc='Validation'):
            text_features = batch_data['text_features'].to(device)
            image_features = batch_data['image_features'].to(device)
            labels = batch_data['label'].to(device)
            
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算验证指标
    val_accuracy = accuracy_score(all_labels, all_predictions)
    val_precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    val_f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, val_accuracy, val_precision, val_recall, val_f1

def main():
    parser = argparse.ArgumentParser(description='训练完整数据集的多模态假新闻检测模型')
    parser.add_argument('--train_csv', type=str, default='/root/autodl-tmp/data/train.csv',
                        help='训练数据CSV文件路径')
    parser.add_argument('--val_csv', type=str, default='/root/autodl-tmp/data/val.csv',
                        help='验证数据CSV文件路径')
    parser.add_argument('--test_csv', type=str, default='/root/autodl-tmp/data/test.csv',
                        help='测试数据CSV文件路径')
    parser.add_argument('--image_dir', type=str, default='/root/autodl-tmp/data/images',
                        help='图像文件目录')
    parser.add_argument('--model_save_path', type=str, 
                        default='/root/models/checkpoints/full_multimodal_model.pth',
                        help='模型保存路径')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=15,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--patience', type=int, default=5,
                        help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # 初始化预处理器
    print("初始化预处理器...")
    preprocessor = FakeNewsDataPreprocessor()
    # 修改为完整数据集路径
    preprocessor.train_data_path = args.train_csv
    preprocessor.val_data_path = args.val_csv
    preprocessor.test_data_path = args.test_csv
    preprocessor.images_dir = args.image_dir
    
    # 重新加载数据集
    print("重新加载完整数据集...")
    preprocessor.train_df = pd.read_csv(args.train_csv, encoding='utf-8')
    preprocessor.val_df = pd.read_csv(args.val_csv, encoding='utf-8')
    preprocessor.test_df = pd.read_csv(args.test_csv, encoding='utf-8')
    
    # 加载和清洗数据
    print("加载训练数据...")
    train_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='train',
        max_samples=None,
        device=device,
        use_cache=False,
        augment_text=False
    )
    train_dataset.df = clean_dataframe(train_dataset.df)
    
    print("加载验证数据...")
    val_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='val',
        max_samples=None,
        device=device,
        use_cache=False,
        augment_text=False
    )
    val_dataset.df = clean_dataframe(val_dataset.df)
    
    print("加载测试数据...")
    test_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='test',
        max_samples=None,
        device=device,
        use_cache=False,
        augment_text=False
    )
    test_dataset.df = clean_dataframe(test_dataset.df)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")
    
    # 初始化模型
    print("初始化模型...")
    model = MultiModalFusionModel(
        text_dim=768,  # BERT特征维度
        image_dim=768,  # CLIP特征维度
        hidden_dim=512,
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # 训练历史记录
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'learning_rate': []
    }
    
    # 早停机制
    best_val_f1 = 0
    patience_counter = 0
    
    print(f"开始训练，共{args.epochs}个epoch...")
    start_time = time.time()
    
    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 更新学习率
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)
        history['val_precision'].append(val_prec)
        history['val_recall'].append(val_rec)
        history['val_f1'].append(val_f1)
        history['learning_rate'].append(current_lr)
        
        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"验证精确率: {val_prec:.4f}, 验证召回率: {val_rec:.4f}, 验证F1: {val_f1:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"验证F1提升到 {val_f1:.4f}，保存模型...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'history': history
            }, args.model_save_path)
        else:
            patience_counter += 1
            print(f"验证F1未提升，耐心计数: {patience_counter}/{args.patience}")
        
        # 早停检查
        if patience_counter >= args.patience:
            print(f"验证F1连续{args.patience}个epoch未提升，提前停止训练")
            break
    
    training_time = time.time() - start_time
    print(f"\n训练完成，总用时: {training_time/3600:.2f}小时")
    
    # 加载最佳模型进行测试
    print("\n加载最佳模型进行测试...")
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试模型
    test_loss, test_acc, test_prec, test_rec, test_f1 = validate_epoch(
        model, test_loader, criterion, device
    )
    
    print(f"\n=== 最终测试结果 ===")
    print(f"测试损失: {test_loss:.4f}")
    print(f"测试准确率: {test_acc:.4f}")
    print(f"测试精确率: {test_prec:.4f}")
    print(f"测试召回率: {test_rec:.4f}")
    print(f"测试F1分数: {test_f1:.4f}")
    
    # 保存训练历史和结果
    history_path = args.model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    results = {
        'model_path': args.model_save_path,
        'training_time_hours': training_time / 3600,
        'best_val_f1': best_val_f1,
        'final_test_results': {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1
        },
        'training_config': {
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'patience': args.patience
        },
        'dataset_sizes': {
            'train': len(train_dataset),
            'val': len(val_dataset),
            'test': len(test_dataset)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = args.model_save_path.replace('.pth', '_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n训练历史保存到: {history_path}")
    print(f"训练结果保存到: {results_path}")
    print(f"模型权重保存到: {args.model_save_path}")

if __name__ == '__main__':
    main()