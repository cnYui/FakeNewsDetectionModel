#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用清洗后数据集训练多模态假新闻检测模型
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np
from tqdm import tqdm

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.MultiModalFakeNewsDetector import (
    FakeNewsDataPreprocessor, 
    FakeNewsDataset, 
    TransformerFusion
)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
    
    for batch_idx, batch_data in enumerate(progress_bar):
        # 处理字典格式的数据
        text_features = batch_data['text_features'].to(device)
        image_features = batch_data['image_features'].to(device)
        labels = batch_data['label'].to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(text_features, image_features)
        loss = criterion(outputs, labels)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # 计算预测结果
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
        })
    
    # 计算epoch指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return avg_loss, accuracy, precision, recall, f1

def validate_epoch(model, dataloader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
        
        for batch_idx, batch_data in enumerate(progress_bar):
            # 处理字典格式的数据
            text_features = batch_data['text_features'].to(device)
            image_features = batch_data['image_features'].to(device)
            labels = batch_data['label'].to(device)
            
            # 前向传播
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # 计算预测结果
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    # 计算epoch指标
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return avg_loss, accuracy, precision, recall, f1, cm

def main():
    parser = argparse.ArgumentParser(description='训练多模态假新闻检测模型（使用清洗后数据）')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data/cleaned',
                       help='清洗后数据目录路径')
    parser.add_argument('--cache_dir', type=str, default='/root/autodl-tmp/data/cache',
                       help='缓存目录路径')
    parser.add_argument('--model_cache_dir', type=str, default='/root/autodl-tmp/model_cache',
                       help='预训练模型缓存目录')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                       help='模型检查点保存目录')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                       help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--save_every', type=int, default=2,
                       help='每隔多少个epoch保存一次模型')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    train_csv = os.path.join(args.data_dir, 'train_cleaned.csv')
    val_csv = os.path.join(args.data_dir, 'val_cleaned.csv')
    test_csv = os.path.join(args.data_dir, 'test_cleaned.csv')
    
    # 检查文件是否存在
    for file_path in [train_csv, val_csv, test_csv]:
        if not os.path.exists(file_path):
            print(f"错误: 找不到文件 {file_path}")
            print("请先运行数据清洗脚本: python data_cleaning.py")
            return
    
    print(f"使用清洗后的数据集:")
    print(f"训练集: {train_csv}")
    print(f"验证集: {val_csv}")
    print(f"测试集: {test_csv}")
    
    # 初始化数据预处理器
    print("\n初始化数据预处理器...")
    preprocessor = FakeNewsDataPreprocessor()
    
    # 重新设置数据路径为清洗后的数据
    print("\n加载清洗后的数据集...")
    preprocessor.train_data_path = train_csv
    preprocessor.val_data_path = val_csv
    preprocessor.test_data_path = test_csv
    
    # 重新加载数据
    preprocessor.train_df = pd.read_csv(train_csv)
    preprocessor.val_df = pd.read_csv(val_csv)
    preprocessor.test_df = pd.read_csv(test_csv)
    
    print(f"清洗后数据统计:")
    print(f"训练集: {len(preprocessor.train_df)} 条")
    print(f"验证集: {len(preprocessor.val_df)} 条")
    print(f"测试集: {len(preprocessor.test_df)} 条")
    
    # 数据已在初始化时加载，无需额外预处理
    print("\n数据已加载完成")
    
    # 创建数据集
    print("\n创建数据集...")
    train_dataset = FakeNewsDataset(preprocessor, split='train', max_samples=len(preprocessor.train_df), device=device)
    val_dataset = FakeNewsDataset(preprocessor, split='val', max_samples=len(preprocessor.val_df), device=device)
    test_dataset = FakeNewsDataset(preprocessor, split='test', max_samples=len(preprocessor.test_df), device=device)
    
    print(f"数据集大小:")
    print(f"训练集: {len(train_dataset)} 条")
    print(f"验证集: {len(val_dataset)} 条")
    print(f"测试集: {len(test_dataset)} 条")
    
    # 创建数据加载器
    print("\n创建数据加载器...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0,  # 避免多进程问题
        pin_memory=False  # 避免CUDA pin_memory问题
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
    
    # 初始化模型
    print("\n初始化模型...")
    model = TransformerFusion(
        text_dim=768,  # DistilBERT特征维度
        image_dim=768,  # 图像特征维度
        hidden_dim=512,
        num_labels=2
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 训练历史记录
    train_history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }
    
    best_val_f1 = 0
    best_epoch = 0
    
    print(f"\n开始训练 (共 {args.epochs} 个epoch)...")
    print("=" * 80)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, val_prec, val_rec, val_f1, val_cm = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录历史
        train_history['train_loss'].append(train_loss)
        train_history['train_acc'].append(train_acc)
        train_history['train_f1'].append(train_f1)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['val_f1'].append(val_f1)
        
        epoch_time = time.time() - epoch_start_time
        
        # 打印epoch结果
        print(f"\nEpoch {epoch+1}/{args.epochs} - 用时: {epoch_time:.2f}s")
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        print(f"验证混淆矩阵:\n{val_cm}")
        
        # 保存最佳模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model_cleaned.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'train_history': train_history
            }, best_model_path)
            
            print(f"✓ 保存最佳模型 (F1: {val_f1:.4f})")
        
        # 定期保存检查点
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_cleaned_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'train_history': train_history
            }, checkpoint_path)
            
            print(f"✓ 保存检查点: {checkpoint_path}")
        
        print("-" * 80)
    
    total_time = time.time() - start_time
    
    print(f"\n训练完成!")
    print(f"总用时: {total_time:.2f}s ({total_time/60:.2f}分钟)")
    print(f"最佳验证F1: {best_val_f1:.4f} (Epoch {best_epoch})")
    
    # 保存训练历史
    history_path = os.path.join(args.checkpoint_dir, 'train_history_cleaned.json')
    with open(history_path, 'w') as f:
        json.dump(train_history, f, indent=2)
    
    print(f"训练历史已保存到: {history_path}")
    
    # 在测试集上评估最佳模型
    print("\n在测试集上评估最佳模型...")
    best_model_path = os.path.join(args.checkpoint_dir, 'best_model_cleaned.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc, test_prec, test_rec, test_f1, test_cm = validate_epoch(
            model, test_loader, criterion, device, -1
        )
        
        print(f"\n测试集结果:")
        print(f"Loss: {test_loss:.4f}")
        print(f"Accuracy: {test_acc:.4f}")
        print(f"Precision: {test_prec:.4f}")
        print(f"Recall: {test_rec:.4f}")
        print(f"F1-Score: {test_f1:.4f}")
        print(f"混淆矩阵:\n{test_cm}")
        
        # 保存测试结果
        test_results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_precision': test_prec,
            'test_recall': test_rec,
            'test_f1': test_f1,
            'confusion_matrix': test_cm.tolist(),
            'best_epoch': best_epoch,
            'best_val_f1': best_val_f1
        }
        
        results_path = os.path.join(args.checkpoint_dir, 'test_results_cleaned.json')
        with open(results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        print(f"测试结果已保存到: {results_path}")

if __name__ == '__main__':
    main()