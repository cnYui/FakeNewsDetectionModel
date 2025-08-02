#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加载预训练文本模型并训练融合模型
"""

import os
import sys
import torch
import logging
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import DistilBertTokenizer, CLIPTokenizer, CLIPProcessor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到系统路径
# 这样可以导入其他模块
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

# 导入我们的模型
from testWordModel.TextProcessingModel import TextProcessingModel, FakeNewsDataset
from testPictureModel.ImageProcessingModel import ImageProcessingModel, FakeNewsImageDataset
from fusionModels.TransformerFusionModel import create_fusion_model
from fusionModels.train_fusion_model import MultiModalDataset, train_model, evaluate_model, plot_training_history, setup_mps_device

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 设置常量
CACHE_DIR = '/Users/wujianxiang/Documents/GitHub/models/model_cache'
DEFAULT_IMAGE_PATH = os.path.join(CACHE_DIR, 'default_image.jpg')

# 预训练模型路径
PRETRAINED_TEXT_MODEL_PATH = '/Users/wujianxiang/Documents/GitHub/models/testWordModel/models/fake_news_model_epoch_2(F1 0.933top).pth'

def load_pretrained_text_model(model_path, device):
    """
    加载预训练的文本模型
    
    Args:
        model_path: 模型路径
        device: 设备
        
    Returns:
        model: 加载的模型
    """
    logger.info(f"正在加载预训练文本模型: {model_path}")
    
    # 创建模型实例
    model = TextProcessingModel(num_classes=2, use_local_models=False)
    
    # 加载模型权重
    try:
        # 检查文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")
            
        # 加载模型权重
        checkpoint = torch.load(model_path, map_location=device)
        
        # 检查是否包含模型状态字典
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"成功加载模型状态字典")
            
            # 打印额外信息（如果有）
            if 'epoch' in checkpoint:
                logger.info(f"模型训练轮数: {checkpoint['epoch']}")
            if 'best_val_f1' in checkpoint:
                logger.info(f"最佳验证F1分数: {checkpoint['best_val_f1']:.4f}")
        else:
            # 尝试直接加载状态字典
            model.load_state_dict(checkpoint)
            logger.info(f"成功加载模型状态字典（直接格式）")
            
        logger.info(f"预训练文本模型加载成功")
    except Exception as e:
        logger.error(f"加载预训练模型失败: {e}")
        raise
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    # 设置为评估模式
    model.eval()
    
    return model

def create_image_model(device):
    """
    创建图像处理模型
    
    Args:
        device: 设备
        
    Returns:
        model: 图像处理模型
    """
    logger.info("创建图像处理模型")
    
    # 创建模型实例
    model = ImageProcessingModel(num_classes=2, use_local_models=True)
    
    # 将模型移动到指定设备
    model = model.to(device)
    
    return model

def prepare_tokenizers_and_processors():
    """
    准备分词器和处理器
    
    Returns:
        bert_tokenizer: BERT分词器
        clip_tokenizer: CLIP分词器
        clip_processor: CLIP处理器
    """
    logger.info("准备分词器和处理器")
    
    try:
        # 强制使用本地模型
        bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
        clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        logger.info("成功从本地加载分词器和处理器")
    except Exception as e:
        logger.error(f"从本地加载分词器和处理器失败: {e}")
        logger.error("请确保已经下载了所需的模型文件")
        raise
    
    return bert_tokenizer, clip_tokenizer, clip_processor

def main(args):
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = setup_mps_device()
    logger.info(f"使用设备: {device}")
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 准备分词器和处理器
    bert_tokenizer, clip_tokenizer, clip_processor = prepare_tokenizers_and_processors()
    
    # 加载预训练的文本模型
    text_model = load_pretrained_text_model(args.text_model_path, device)
    
    # 创建图像处理模型
    image_model = create_image_model(device)
    
    # 创建融合模型
    fusion_model = create_fusion_model(text_model, image_model, fusion_dim=512, num_classes=2)
    fusion_model = fusion_model.to(device)
    logger.info("融合模型创建成功")
    
    # 创建数据集
    train_dataset = MultiModalDataset(
        data_path=args.train_data_path,
        images_dir=args.images_dir,
        bert_tokenizer=bert_tokenizer,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        max_samples=args.max_samples
    )
    
    val_dataset = MultiModalDataset(
        data_path=args.val_data_path,
        images_dir=args.images_dir,
        bert_tokenizer=bert_tokenizer,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        max_samples=args.max_samples // 5 if args.max_samples else None
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    logger.info(f"训练集大小: {len(train_dataset)}")
    logger.info(f"验证集大小: {len(val_dataset)}")
    
    # 训练融合模型
    trained_model, history = train_model(
        model=fusion_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_path=args.save_path,
        patience=args.patience
    )
    
    # 绘制训练历史
    plot_training_history(history)
    
    # 评估模型
    metrics = evaluate_model(trained_model, val_loader, device)
    logger.info(f"最终评估结果: {metrics}")
    
    logger.info("训练完成")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='加载预训练文本模型并训练融合模型')
    
    # 模型路径
    parser.add_argument('--text_model_path', type=str, default=PRETRAINED_TEXT_MODEL_PATH,
                        help='预训练文本模型路径')
    parser.add_argument('--save_path', type=str, default='fusion_model.pth',
                        help='融合模型保存路径')
    
    # 数据路径
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='训练数据路径')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='验证数据路径')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='图像目录路径')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='梯度累积步数')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值')
    
    # 其他参数
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数，用于限制内存使用')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作进程数')
    
    args = parser.parse_args()
    
    main(args)
