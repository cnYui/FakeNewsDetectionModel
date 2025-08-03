#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多模态假新闻检测模型训练脚本
完整的训练流程，包括数据加载、模型训练、验证和保存
"""

import os
import sys
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import logging
import argparse
from datetime import datetime
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# 添加项目路径
sys.path.append('/root/models')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def evaluate_model(model, dataloader, criterion, device):
    """
    评估模型性能
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            text_features = batch['text_features'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_predictions, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_model(args):
    """
    训练模型主函数
    """
    logger.info("开始训练多模态假新闻检测模型...")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 导入必要模块
    from src.models.MultiModalFakeNewsDetector import FakeNewsDataPreprocessor, FakeNewsDataset
    
    # 初始化数据预处理器
    logger.info("初始化数据预处理器...")
    preprocessor = FakeNewsDataPreprocessor()
    
    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='train',
        max_samples=args.max_samples,
        device=device,
        use_cache=False,
        augment_text=args.augment_text
    )
    
    val_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='val',
        max_samples=args.max_samples // 5 if args.max_samples else None,
        device=device,
        use_cache=False,
        augment_text=False
    )
    
    test_dataset = FakeNewsDataset(
        preprocessor=preprocessor,
        split='test',
        max_samples=args.max_samples // 5 if args.max_samples else None,
        device=device,
        use_cache=False,
        augment_text=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    logger.info(f"测试集大小: {len(test_dataset)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = MultiModalFusionModel(
        text_dim=768,
        image_dim=768,
        hidden_dim=args.hidden_dim,
        num_classes=2,
        dropout=args.dropout
    ).to(device)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # 训练循环
    logger.info("开始训练...")
    best_val_f1 = 0
    train_history = []
    
    for epoch in range(args.epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0
        train_predictions = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            text_features = batch['text_features'].to(device)
            image_features = batch['image_features'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(text_features, image_features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_predictions.extend(predictions.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{args.epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # 计算训练指标
        avg_train_loss = total_train_loss / len(train_loader)
        train_accuracy = accuracy_score(train_labels, train_predictions)
        
        # 验证阶段
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics['loss'])
        
        # 记录训练历史
        epoch_history = {
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        train_history.append(epoch_history)
        
        logger.info(f"Epoch {epoch+1}/{args.epochs}:")
        logger.info(f"  训练损失: {avg_train_loss:.4f}, 训练准确率: {train_accuracy:.4f}")
        logger.info(f"  验证损失: {val_metrics['loss']:.4f}, 验证准确率: {val_metrics['accuracy']:.4f}")
        logger.info(f"  验证F1: {val_metrics['f1']:.4f}, 学习率: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'args': vars(args)
            }, args.model_save_path)
            logger.info(f"保存最佳模型，验证F1: {best_val_f1:.4f}")
    
    # 加载最佳模型进行测试
    logger.info("加载最佳模型进行测试...")
    checkpoint = torch.load(args.model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 测试阶段
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    logger.info("测试结果:")
    logger.info(f"  测试损失: {test_metrics['loss']:.4f}")
    logger.info(f"  测试准确率: {test_metrics['accuracy']:.4f}")
    logger.info(f"  测试精确率: {test_metrics['precision']:.4f}")
    logger.info(f"  测试召回率: {test_metrics['recall']:.4f}")
    logger.info(f"  测试F1: {test_metrics['f1']:.4f}")
    
    # 保存训练历史
    history_path = args.model_save_path.replace('.pth', '_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(train_history, f, indent=2, ensure_ascii=False)
    
    # 保存最终结果
    results = {
        'best_val_f1': best_val_f1,
        'test_metrics': test_metrics,
        'args': vars(args),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = args.model_save_path.replace('.pth', '_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练完成！最佳验证F1: {best_val_f1:.4f}, 测试F1: {test_metrics['f1']:.4f}")
    logger.info(f"模型保存路径: {args.model_save_path}")
    logger.info(f"训练历史保存路径: {history_path}")
    logger.info(f"结果保存路径: {results_path}")

def main():
    parser = argparse.ArgumentParser(description='多模态假新闻检测模型训练')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--hidden_dim', type=int, default=512, help='隐藏层维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--max_samples', type=int, default=None, help='最大样本数（用于调试）')
    parser.add_argument('--augment_text', action='store_true', help='是否启用文本增强')
    parser.add_argument('--model_save_path', type=str, default='/root/models/checkpoints/multimodal_model.pth', help='模型保存路径')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(os.path.dirname(args.model_save_path), exist_ok=True)
    
    # 开始训练
    train_model(args)

if __name__ == "__main__":
    main()