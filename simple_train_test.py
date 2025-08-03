#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练测试脚本
用于验证多模态假新闻检测系统的基本训练流程
"""

import os
import sys
import torch
import pandas as pd
from torch.utils.data import DataLoader
import logging

# 添加项目路径
sys.path.append('/root/models')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_training():
    """
    测试基本训练流程
    """
    try:
        logger.info("开始基本训练测试...")
        
        # 1. 导入必要模块
        logger.info("导入模块...")
        from src.models.MultiModalFakeNewsDetector import FakeNewsDataPreprocessor, FakeNewsDataset
        from src.models.text_models.TextProcessingModel import TextProcessingModel
        from src.models.image_models.ImageProcessingModel import ImageProcessingModel
        from src.models.TransformerFusionModel import create_fusion_model
        
        # 2. 初始化数据预处理器
        logger.info("初始化数据预处理器...")
        preprocessor = FakeNewsDataPreprocessor()
        
        # 3. 检查预处理器中的数据
        logger.info("检查预处理器数据...")
        logger.info(f"训练数据大小: {len(preprocessor.train_df)}")
        logger.info(f"验证数据大小: {len(preprocessor.val_df)}")
        logger.info(f"测试数据大小: {len(preprocessor.test_df)}")
        
        # 检查数据类型
        sample_text = preprocessor.train_df.iloc[0]['text']
        logger.info(f"样本文本类型: {type(sample_text)}")
        logger.info(f"样本文本内容: {str(sample_text)[:100]}...")
        
        # 4. 创建数据集
        logger.info("创建数据集...")
        train_dataset = FakeNewsDataset(
            preprocessor=preprocessor,
            split='dev',
            max_samples=50,  # 使用小样本进行测试
            device='cuda' if torch.cuda.is_available() else 'cpu',
            use_cache=False,  # 禁用缓存以避免问题
            augment_text=False
        )
        
        # 5. 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=2,  # 使用小批量进行测试
            shuffle=True,
            num_workers=0  # 避免多进程问题
        )
        
        # 6. 测试数据加载
        logger.info("测试数据加载...")
        for i, batch in enumerate(train_loader):
            if i >= 2:  # 只测试前2个批次
                break
            logger.info(f"批次 {i+1}:")
            logger.info(f"  - text_features: {batch['text_features'].shape}")
            logger.info(f"  - image_features: {batch['image_features'].shape}")
            logger.info(f"  - label: {batch['label'].shape}")
        
        # 7. 创建简化的融合模型
        logger.info("创建简化融合模型...")
        
        class SimpleFusionModel(torch.nn.Module):
            def __init__(self, text_dim=768, image_dim=768, hidden_dim=256, num_classes=2):
                super().__init__()
                self.text_proj = torch.nn.Linear(text_dim, hidden_dim)
                self.image_proj = torch.nn.Linear(image_dim, hidden_dim)
                self.fusion = torch.nn.Linear(hidden_dim * 2, hidden_dim)
                self.classifier = torch.nn.Linear(hidden_dim, num_classes)
                self.dropout = torch.nn.Dropout(0.1)
                
            def forward(self, text_features, image_features):
                # 投影到相同维度
                text_proj = self.text_proj(text_features.squeeze(1))  # 去掉batch维度
                image_proj = self.image_proj(image_features.squeeze(1))
                
                # 融合特征
                fused = torch.cat([text_proj, image_proj], dim=1)
                fused = self.fusion(fused)
                fused = torch.relu(fused)
                fused = self.dropout(fused)
                
                # 分类
                logits = self.classifier(fused)
                return logits
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        fusion_model = SimpleFusionModel().to(device)
        logger.info(f"模型已移动到设备: {device}")
        
        # 8. 测试模型前向传播
        logger.info("测试模型前向传播...")
        fusion_model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(train_loader):
                if i >= 1:  # 只测试1个批次
                    break
                
                outputs = fusion_model(
                    batch['text_features'],
                    batch['image_features']
                )
                
                logger.info(f"模型输出形状: {outputs.shape}")
                logger.info(f"预测概率: {torch.softmax(outputs, dim=1)}")
                break
        
        # 9. 测试简单训练步骤
        logger.info("测试训练步骤...")
        fusion_model.train()
        optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-4)
        criterion = torch.nn.CrossEntropyLoss()
        
        for i, batch in enumerate(train_loader):
            if i >= 2:  # 只训练2个批次
                break
            
            optimizer.zero_grad()
            
            outputs = fusion_model(
                batch['text_features'],
                batch['image_features']
            )
            
            loss = criterion(outputs, batch['label'])
            loss.backward()
            optimizer.step()
            
            logger.info(f"批次 {i+1}, 损失: {loss.item():.4f}")
        
        logger.info("✓ 基本训练测试完成！")
        return True
        
    except Exception as e:
        logger.error(f"训练测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_training()
    if success:
        print("\n🎉 基本训练测试成功！系统可以进行训练。")
    else:
        print("\n❌ 基本训练测试失败，需要进一步调试。")