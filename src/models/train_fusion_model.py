#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
u591au6a21u6001u878du5408u6a21u578bu8badu7ec3u811au672c
u52a0u8f7du6587u672cu5904u7406u6a21u578bu548cu56feu50cfu5904u7406u6a21u578bu548cu8badu7ec3
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import sys
import json
import time
import logging
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import random
from datetime import datetime
import re
import csv
from torchvision import transforms

# u6dfbu52a0u9879u76eeu6839u76eeu5f55u5230u7cfbu7edfu8defu5f84
# u8fd9u6837u53efu4ee5u5bfcu5165u5176u4ed6u6a21u5757
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

# 导入我们的模型
from src.models.text_models.TextProcessingModel import TextProcessingModel, FakeNewsDataset
from src.models.image_models.ImageProcessingModel import ImageProcessingModel, FakeNewsImageDataset
from src.models.TransformerFusionModel import create_fusion_model

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
CACHE_DIR = '/root/autodl-tmp/model_cache_new'
DEFAULT_IMAGE_PATH = os.path.join(CACHE_DIR, 'default_image.jpg')

# 设置MPS设备（适用于Apple Silicon）
def setup_mps_device():
    """
    u8bbeu7f6eMPSu8bbeu5907u5e76u5904u7406u53efu80fdu7684u517cu5bb9u6027u95eeu9898
    
    Returns:
        device: u53efu7528u7684u8bbeu5907uff08mpsu3001cudau6216cpuuff09
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"u4f7fu7528MPSu8bbeu5907: {device}")
        
        # u6d4bu8bd5MPSu517cu5bb9u6027
        try:
            # u521bu5efau5c0fu578bu5f20u91cfu6d4bu8bd5
            x = torch.ones(1, 1).to(device)
            y = x + 1
            assert torch.all(y == 2), "u57fau672cu8fd0u7b97u6d4bu8bd5u5931u8d25"
            print("u57fau672cu8fd0u7b97u6d4bu8bd5u6210u529f")
        except Exception as e:
            print(f"MPSu517cu5bb9u6027u6d4bu8bd5u5931u8d25: {e}")
            print("u56deu9000u5230CPUu8bbeu5907")
            device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"u4f7fu7528CUDAu8bbeu5907: {device}")
    else:
        device = torch.device("cpu")
        print("u4f7fu7528CPUu8bbeu5907")
    
    return device

# u591au6a21u6001u6570u636eu96c6u7c7b
class MultiModalDataset(Dataset):
    """
    u591au6a21u6001u6570u636eu96c6u7c7buff0cu7ed3u5408u6587u672cu548cu56feu50cfu6570u636e
    """
    def __init__(self, data_path, images_dir, bert_tokenizer, clip_tokenizer, clip_processor, transform=None,
                 max_length=512, clip_max_length=77, image_size=(224, 224), max_samples=None):
        """
        u521du59cbu5316u6570u636eu96c6
        
        Args:
            data_path: u6570u636eu6587u4ef6u8defu5f84
            images_dir: u56feu50cfu76eeu5f55u8defu5f84
            bert_tokenizer: BERTu5206u8bcdu5668
            clip_tokenizer: CLIPu5206u8bcdu5668
            clip_processor: CLIPu5904u7406u5668
            transform: u56feu50cfu53d8u6362uff08u7528u4e8eResNetuff09
            max_length: BERTu6700u5927u5e8fu5217u957fu5ea6
            clip_max_length: CLIPu6700u5927u5e8fu5217u957fu5ea6
            image_size: u56feu50cfu5c3au5bf8
            max_samples: u6700u5927u6837u672cu6570
        """
        self.data_path = data_path
        self.images_dir = images_dir
        self.bert_tokenizer = bert_tokenizer
        self.clip_tokenizer = clip_tokenizer
        self.clip_processor = clip_processor
        self.transform = transform
        self.max_length = max_length
        self.clip_max_length = clip_max_length
        self.image_size = image_size
        
        # u52a0u8f7du6570u636e
        try:
            # u68c0u67e5u6587u4ef6u683cu5f0f
            with open(data_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if ',' in first_line and '\t' not in first_line:
                    # CSVu683cu5f0f
                    print(f"u68c0u6d4bu5230CSVu683cu5f0fu6570u636eu6587u4ef6")
                    # u5148u5c1du8bd5u6b63u5e38u8bfbu53d6
                    self.df = pd.read_csv(
                        data_path,
                        encoding='utf-8'
                    )
                    
                    # u68c0u67e5u662fu5426u9700u8981u62c6u5206u5217
                    if len(self.df.columns) == 1 and ',' in self.df.columns[0]:
                        column_names = self.df.columns[0].split(',')
                        print(f"u68c0u6d4bu5230u9700u8981u62c6u5206u7684u5217u540d: {column_names}")
                        
                        # u91cdu65b0u52a0u8f7du6570u636euff0cu6b63u786eu6307u5b9au5217u540d
                        self.df = pd.read_csv(
                            data_path,
                            names=column_names,
                            header=0,
                            encoding='utf-8'
                        )
                else:
                    # TSVu683cu5f0f
                    print(f"u68c0u6d4bu5230TSVu683cu5f0fu6570u636eu6587u4ef6")
                    self.df = pd.read_csv(
                        data_path,
                        delimiter='\t',
                        quoting=3,  # QUOTE_NONE
                        encoding='utf-8',
                        escapechar='\\',
                        engine='python'  # u4f7fu7528Pythonu5f15u64ceu89e3u6790uff0cu89e3u51b3Cu89e3u6790u5668u7684u9519u8bef
                    )
            
            print(f"u6570u636eu96c6u5217u540d: {list(self.df.columns)}")
            print(f"u6570u636eu96c6u52a0u8f7du5b8cu6210uff0cu5171u6709{len(self.df)}u6761u8bb0u5f55")
            
            # u9650u5236u6837u672cu6570
            if max_samples is not None and max_samples > 0 and len(self.df) > max_samples:
                self.df = self.df.sample(max_samples, random_state=42).reset_index(drop=True)
                print(f"u9650u5236u6837u672cu6570u4e3a {max_samples}")
        except Exception as e:
            print(f"u52a0u8f7du6570u636eu96c6u65f6u51fau9519: {str(e)}")
            raise
        
        # u68c0u67e5u5e76u5904u7406u56feu50cfIDu5217
        self._process_image_column()
        
        # u5c6u6807u7b7eu8f6cu6362u4e3au6570u5b57
        self._process_label_column()
        
        # u521bu5efau9ed8u8ba4u56feu50cf
        self.default_image = self._create_default_image()
        
        # u9a8cu8bc1u6709u6548u6837u672c
        self._validate_samples()
    
    def _get_image_path(self, idx):
        """
        获取图像路径
        
        Args:
            idx: 数据索引
        
        Returns:
            图像路径或None（如果找不到图像）
        """
        if self.image_column is None:
            return None
        
        # 获取图像ID
        image_id_value = self.df.iloc[idx][self.image_column]
        
        # 处理可能的多个图像ID（以逗号分隔）
        if isinstance(image_id_value, str) and ',' in image_id_value:
            image_ids = [id.strip() for id in image_id_value.split(',')]
            # 使用第一个有效的图像ID
            for image_id in image_ids:
                # 尝试不同的文件扩展名
                for ext in ['.jpg', '.jpeg', '.png']:
                    img_path = os.path.join(self.images_dir, f"{image_id}{ext}")
                    if os.path.exists(img_path):
                        return img_path
            # 如果所有图像ID都无效，返回None
            return None
        elif isinstance(image_id_value, str):
            # 单个图像ID
            image_id = image_id_value.strip()
            # 尝试不同的文件扩展名
            for ext in ['.jpg', '.jpeg', '.png']:
                img_path = os.path.join(self.images_dir, f"{image_id}{ext}")
                if os.path.exists(img_path):
                    return img_path
            # 如果图像ID无效，返回None
            return None
        else:
            # 如果图像ID不是字符串，返回None
            return None
    
    def _convert_label_to_int(self, label):
        """
        将标签转换为整数
        
        Args:
            label: 原始标签（可能是字符串或数字）
            
        Returns:
            整数标签（0表示真新闻，1表示假新闻）
        """
        # 如果标签已经是数字，直接返回
        if isinstance(label, (int, float)):
            return int(label)
        
        # 如果标签是字符串，进行转换
        if isinstance(label, str):
            # 将标签转换为小写并去除空格
            label = label.lower().strip()
            
            # 处理各种可能的标签格式
            if label in ['fake', 'false', 'f', '1', 'y', 'yes', 'fake news', 'rumor']:
                return 1  # 假新闻
            elif label in ['real', 'true', 't', '0', 'n', 'no', 'real news', 'non-rumor']:
                return 0  # 真新闻
            else:
                # 如果标签不在预期的值中，尝试将其转换为整数
                try:
                    return int(label)
                except ValueError:
                    print(f"警告: 无法转换标签 '{label}' 为整数，使用默认值 0")
                    return 0  # 默认为真新闻
        
        # 如果标签是 None 或其他类型，返回默认值
        print(f"警告: 无法转换标签类型 {type(label)}，使用默认值 0")
        return 0  # 默认为真新闻
    
    def __len__(self):
        """
        返回数据集长度
        """
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含所有模型输入和标签的字典
        """
        # 获取有效索引
        if self.valid_indices and idx < len(self.valid_indices):
            real_idx = self.valid_indices[idx]
        else:
            real_idx = idx
            
        # 获取文本
        text = self.df.iloc[real_idx]['post_text'] if 'post_text' in self.df.columns else ""
        
        # 获取图像路径
        image_path = self.image_paths[real_idx]
        
        # 处理文本
        bert_inputs = self.bert_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        bert_input_ids = bert_inputs["input_ids"].squeeze(0)
        bert_attention_mask = bert_inputs["attention_mask"].squeeze(0)
        
        clip_text_inputs = self.clip_processor.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=77)
        clip_input_ids = clip_text_inputs["input_ids"].squeeze(0)
        clip_attention_mask = clip_text_inputs["attention_mask"].squeeze(0)
        
        # 处理图像
        try:
            # 尝试加载图像
            if os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
                
                # 调整图像大小
                if self.image_size:
                    image = image.resize(self.image_size)
                
                # 应用变换
                if self.transform:
                    resnet_image = self.transform(image)
                else:
                    # 默认变换
                    resnet_image = transforms.ToTensor()(image)
                    resnet_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(resnet_image)
                
                # 处理CLIP图像
                clip_pixel_values = self.clip_processor.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
            else:
                # 如果图像不存在，使用默认图像
                print(f"警告: 图像不存在: {image_path}，使用默认图像")
                resnet_image, clip_pixel_values = self._create_default_image()
        except Exception as e:
            print(f"处理图像时出错: {e}, 路径: {image_path}")
            resnet_image, clip_pixel_values = self._create_default_image()
        
        # 获取标签
        label = self.df.iloc[real_idx]['label_num']
        
        return {
            "bert_input_ids": bert_input_ids,
            "bert_attention_mask": bert_attention_mask,
            "clip_input_ids": clip_input_ids,
            "clip_attention_mask": clip_attention_mask,
            "resnet_image": resnet_image,
            "clip_pixel_values": clip_pixel_values,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _create_default_image(self, size=(224, 224)):
        """
        创建默认图像（纯黑色）
        
        Args:
            size: 图像大小
            
        Returns:
            resnet_image: 用于ResNet的图像张量
            clip_pixel_values: 用于CLIP的图像张量
        """
        # 创建黑色图像
        image = Image.new('RGB', size, color='black')
        
        # 处理ResNet图像
        if self.transform:
            resnet_image = self.transform(image)
        else:
            # 默认变换
            resnet_image = transforms.ToTensor()(image)
            resnet_image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(resnet_image)
        
        # 处理CLIP图像
        try:
            clip_pixel_values = self.clip_processor.image_processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        except Exception as e:
            print(f"处理默认CLIP图像时出错: {e}")
            # 创建零张量作为CLIP输入
            clip_pixel_values = torch.zeros((3, 224, 224))
        
        return resnet_image, clip_pixel_values

    def _process_image_column(self):
        """处理图像ID列"""
        self.image_column = None
        
        # 检查可能的图像列名
        possible_image_columns = [
            'image', 'images', 'img', 'imgs', 'photo', 'photos', 
            'image_id', 'image_ids', 'img_id', 'img_ids', 'image_path', 'image_paths',
            'image_id(s)', # Twitter数据集特有的列名
            'path'  # 添加'path'作为可能的图像列名
        ]
        
        for col in possible_image_columns:
            if col in self.df.columns:
                self.image_column = col
                print(f"使用 {col} 作为图像列")
                break
        else:
            print(f"警告: 找不到图像列，数据集列名: {list(self.df.columns)}")
            raise ValueError("找不到图像列，请检查数据集格式")
        
        # 处理图像路径
        self.image_paths = []
        for idx, row in self.df.iterrows():
            image_id = row[self.image_column]
            
            # 处理可能有多个图像ID的情况（用逗号分隔）
            if isinstance(image_id, str) and ',' in image_id:
                # 取第一个图像ID
                image_id = image_id.split(',')[0].strip()
                print(f"多图像ID，使用第一个: {image_id}")
            
            # 构建完整的图像路径
            if self.image_column == 'path':
                # 处理相对路径，转换为绝对路径
                if isinstance(image_id, str) and image_id.startswith('./data/'):
                    # 将相对路径转换为绝对路径
                    base_dir = '/Users/wujianxiang/Documents/GitHub/models'
                    image_path = os.path.join(base_dir, image_id[2:])  # 去掉开头的 './'
                else:
                    # 其他情况，使用原始路径
                    image_path = os.path.join(self.images_dir, os.path.basename(image_id))
            else:
                image_path = os.path.join(self.images_dir, f"{image_id}.jpg")
            
            # 检查文件是否存在，尝试不同的扩展名
            if not os.path.exists(image_path):
                for ext in ['.png', '.jpeg', '.JPG', '.JPEG', '.PNG']:
                    alt_path = os.path.join(self.images_dir, f"{image_id}{ext}")
                    if os.path.exists(alt_path):
                        image_path = alt_path
                        break
            
            self.image_paths.append(image_path)
            
            # 只打印前几个路径用于调试
            if idx < 5:
                print(f"图像ID: {image_id}, 路径: {image_path}, 存在: {os.path.exists(image_path)}")
    
    def _process_label_column(self):
        """处理标签列"""
        # 检查可能的标签列名
        if 'label' in self.df.columns:
            self.label_column = 'label'
        else:
            # 尝试找到可能的标签列
            possible_label_columns = ['label', 'Label', 'class', 'Class', 'fake', 'Fake', 'real', 'Real']
            for col in possible_label_columns:
                if col in self.df.columns:
                    self.label_column = col
                    print(f"使用 {col} 作为标签列")
                    break
            else:
                print(f"警告: 找不到标签列，数据集列名: {list(self.df.columns)}")
                raise ValueError("找不到标签列，请检查数据集格式")
        
        # 处理标签
        print(f"原始标签值: {self.df[self.label_column].iloc[:5].tolist()}")
        self.df['label_num'] = self.df[self.label_column].apply(self._convert_label_to_int)
        print(f"转换后的标签值: {self.df['label_num'].iloc[:5].tolist()}")
    
    def _validate_samples(self):
        """
        验证样本
        """
        self.valid_indices = []
        
        # 检查每个样本的图像是否存在
        for idx in range(len(self.df)):
            # 获取图像路径
            image_path = self.image_paths[idx]
            
            # 如果图像存在，则添加到有效索引列表
            if image_path and os.path.exists(image_path):
                self.valid_indices.append(idx)
        
        print(f"有效样本数: {len(self.valid_indices)}")
        
        # 如果没有有效样本，则使用所有样本
        if len(self.valid_indices) == 0:
            print("警告: 没有有效样本，使用所有样本")
            self.valid_indices = list(range(len(self.df)))

# 训练函数
def train_model(model, train_loader, val_loader, device, num_epochs=10, lr=2e-5, weight_decay=1e-4, 
               gradient_accumulation_steps=1, save_path='fusion_model.pth', patience=3):
    """
    训练融合模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        gradient_accumulation_steps: 梯度累积步数
        save_path: 模型保存路径
        patience: 早停耐心值
        
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    print("进入训练函数...")
    # 将模型移动到设备
    model = model.to(device)
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 使用交叉熵损失
    criterion = nn.CrossEntropyLoss()
    
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # 最佳验证损失和早停计数器
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\n开始第 {epoch+1}/{num_epochs} 轮训练")
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # 使用tqdm显示进度条
        progress_bar = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}")
        
        # 梯度累积变量
        optimizer.zero_grad()
        accumulated_steps = 0
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 将数据移动到设备
                bert_input_ids = batch['bert_input_ids'].to(device)
                bert_attention_mask = batch['bert_attention_mask'].to(device)
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                resnet_image = batch['resnet_image'].to(device)
                clip_pixel_values = batch['clip_pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(
                    bert_input_ids=bert_input_ids,
                    bert_attention_mask=bert_attention_mask,
                    clip_input_ids=clip_input_ids,
                    clip_attention_mask=clip_attention_mask,
                    resnet_image=resnet_image,
                    clip_pixel_values=clip_pixel_values
                )
                
                # u5982u679cu8f93u51fau662fu5143u7ec4uff0cu53d6u7b2cu4e00u4e2au5143u7d20uff08logitsuff09
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # u8ba1u7b97u635fu5931
                loss = criterion(logits, labels)
                loss = loss / gradient_accumulation_steps  # 缩放损失
                
                # 反向传播
                loss.backward()
                
                # 更新训练损失
                train_loss += loss.item() * gradient_accumulation_steps
                train_steps += 1
                
                # 梯度累积
                accumulated_steps += 1
                if accumulated_steps % gradient_accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # 更新参数
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': train_loss / train_steps})
            
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 确保最后一个批次的梯度被应用
        if accumulated_steps % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # 计算平均训练损失
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_preds = []
        all_labels = []
        
        print("开始验证...")
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                try:
                    # 将数据移动到设备
                    bert_input_ids = batch['bert_input_ids'].to(device)
                    bert_attention_mask = batch['bert_attention_mask'].to(device)
                    clip_input_ids = batch['clip_input_ids'].to(device)
                    clip_attention_mask = batch['clip_attention_mask'].to(device)
                    resnet_image = batch['resnet_image'].to(device)
                    clip_pixel_values = batch['clip_pixel_values'].to(device)
                    labels = batch['label'].to(device)
                    
                    # 前向传播
                    outputs = model(
                        bert_input_ids=bert_input_ids,
                        bert_attention_mask=bert_attention_mask,
                        clip_input_ids=clip_input_ids,
                        clip_attention_mask=clip_attention_mask,
                        resnet_image=resnet_image,
                        clip_pixel_values=clip_pixel_values
                    )
                    
                    # u5982u679cu8f93u51fau662fu5143u7ec4uff0cu53d6u7b2cu4e00u4e2au5143u7d20uff08logitsuff09
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # u8ba1u7b97u635fu5931
                    loss = criterion(logits, labels)
                    
                    # 更新验证损失
                    val_loss += loss.item()
                    val_steps += 1
                    
                    # 获取预测
                    _, preds = torch.max(logits, dim=1)
                    
                    # 收集预测和标签
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                except Exception as e:
                    print(f"验证出错: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # 计算平均验证损失
        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)
        
        # 计算指标
        val_accuracy = accuracy_score(all_labels, all_preds)
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        # 更新历史
        history['val_accuracy'].append(val_accuracy)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # 打印结果
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  训练损失: {avg_train_loss:.4f}")
        print(f"  验证损失: {avg_val_loss:.4f}")
        print(f"  验证准确率: {val_accuracy:.4f}")
        print(f"  验证精确率: {val_precision:.4f}")
        print(f"  验证召回率: {val_recall:.4f}")
        print(f"  验证F1值: {val_f1:.4f}")
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            print(f"验证损失从 {best_val_loss:.4f} 改善到 {avg_val_loss:.4f}，保存模型...")
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stop_counter = 0
            
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'history': history
            }, save_path)
            print(f"模型已保存到 {save_path}")
        else:
            early_stop_counter += 1
            print(f"验证损失未改善，早停计数器: {early_stop_counter}/{patience}")
        
        # 早停
        if early_stop_counter >= patience:
            print(f"早停触发，停止训练")
            break
    
    # 恢复最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history

# 评估函数
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
    print("开始评估模型...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="评估"):
            try:
                # 将数据移动到设备
                bert_input_ids = batch['bert_input_ids'].to(device)
                bert_attention_mask = batch['bert_attention_mask'].to(device)
                clip_input_ids = batch['clip_input_ids'].to(device)
                clip_attention_mask = batch['clip_attention_mask'].to(device)
                resnet_image = batch['resnet_image'].to(device)
                clip_pixel_values = batch['clip_pixel_values'].to(device)
                labels = batch['label'].to(device)
                
                # 前向传播
                outputs = model(
                    bert_input_ids=bert_input_ids,
                    bert_attention_mask=bert_attention_mask,
                    clip_input_ids=clip_input_ids,
                    clip_attention_mask=clip_attention_mask,
                    resnet_image=resnet_image,
                    clip_pixel_values=clip_pixel_values
                )
                
                # u5982u679cu8f93u51fau662fu5143u7ec4uff0cu53d6u7b2cu4e00u4e2au5143u7d20uff08logitsuff09
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # 获取预测
                _, preds = torch.max(logits, dim=1)
                
                # 收集预测和标签
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            except Exception as e:
                print(f"评估出错: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # 打印结果
    print("\n评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1值: {f1:.4f}")
    print("\n混淆矩阵:")
    print(conf_matrix)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join('fusionModels', 'confusion_matrix.png'))
    plt.close()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix
    }

# 绘制训练历史
def plot_training_history(history):
    """
    绘制训练历史
    
    Args:
        history: 训练历史记录
    """
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 绘制损失
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率
    plt.subplot(2, 2, 2)
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制精确率和召回率
    plt.subplot(2, 2, 3)
    plt.plot(history['val_precision'], label='Precision')
    plt.plot(history['val_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1值
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Validation F1 Score')
    plt.legend()
    plt.grid(True)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join('fusionModels', 'training_history.png'))
    plt.close()

# 主函数
def main(args):
    """
    主函数
    
    Args:
        args: 命令行参数
    """
    # 设置设备
    device = setup_mps_device()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # 设置数据路径
    data_dir = '/Users/wujianxiang/Documents/GitHub/models/data'  # 数据目录
    train_path = '/Users/wujianxiang/Documents/GitHub/models/data/train.csv'  # 训练数据文件
    val_path = '/Users/wujianxiang/Documents/GitHub/models/data/val.csv'  # 验证数据文件
    test_path = '/Users/wujianxiang/Documents/GitHub/models/data/test.csv'  # 测试数据文件
    images_dir = '/Users/wujianxiang/Documents/GitHub/models/data/images'  # 图像文件夹
    test_images_dir = '/Users/wujianxiang/Documents/GitHub/models/data/images'  # Use the same images directory for testing
    
    # 设置模型路径
    text_model_path = args.text_model_path
    image_model_path = args.image_model_path
    fusion_model_path = args.fusion_model_path
    
    # 加载文本模型
    print("加载文本处理模型...")
    from transformers import DistilBertTokenizer, CLIPTokenizer
    
    # 加载分词器
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    
    # 加载文本模型
    text_model = TextProcessingModel(use_local_models=True)
    if os.path.exists(text_model_path):
        print(f"从 {text_model_path} 加载文本模型权重")
        text_model.load_state_dict(torch.load(text_model_path, map_location=device))
    text_model.eval()  # 设置为评估模式
    
    # 加载图像模型
    print("加载图像处理模型...")
    from transformers import CLIPProcessor
    
    # 加载CLIP处理器
    try:
        # 首先尝试从本地路径加载
        clip_processor_path = os.path.join(args.model_cache_dir, 'clip-vit-base-patch32')
        if os.path.exists(clip_processor_path):
            print(f"尝试从本地路径加载CLIP处理器: {clip_processor_path}")
            clip_processor = CLIPProcessor.from_pretrained(clip_processor_path, local_files_only=True)
            print("成功从本地路径加载CLIP处理器")
        else:
            # 如果本地路径不存在，尝试从预训练模型加载
            print("尝试从预训练模型加载CLIP处理器: openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=args.model_cache_dir)
            print("成功从预训练模型加载CLIP处理器")
    except Exception as e:
        print(f"加载CLIP处理器失败: {e}")
        print("创建默认的CLIP处理器")
        # 使用默认的处理器
        from transformers import CLIPImageProcessor, CLIPTokenizer
        clip_processor = CLIPProcessor(
            tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True),
            image_processor=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        )
    
    # 加载图像模型
    image_model = ImageProcessingModel(use_local_models=True, fast_mode=True)  # 使用fast_mode=True加载ResNet-18
    if os.path.exists(image_model_path):
        print(f"从 {image_model_path} 加载图像模型权重")
        image_model.load_state_dict(torch.load(image_model_path, map_location=device))
    image_model.eval()  # 设置为评估模式
    
    # 创建数据集
    print("创建数据集...")
    
    # 如果train_path和val_path相同，则进行训练/验证集分割
    if train_path == val_path and args.train_val_split > 0:
        print(f"使用相同的数据文件进行训练和验证，将按照 {args.train_val_split:.2f}/{1-args.train_val_split:.2f} 的比例分割")
        full_dataset = MultiModalDataset(
            data_path=train_path,
            images_dir=images_dir,
            bert_tokenizer=bert_tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_processor=clip_processor,
            max_samples=args.max_samples
        )
        
        # 计算分割点
        train_size = int(len(full_dataset) * args.train_val_split)
        val_size = len(full_dataset) - train_size
        
        # 随机分割数据集
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        print(f"数据集分割完成：训练集 {len(train_dataset)} 样本，验证集 {len(val_dataset)} 样本")
    else:
        # 使用不同的文件创建训练集和验证集
        train_dataset = MultiModalDataset(
            data_path=train_path,
            images_dir=images_dir,
            bert_tokenizer=bert_tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_processor=clip_processor,
            max_samples=args.max_samples
        )
        
        val_dataset = MultiModalDataset(
            data_path=val_path,
            images_dir=images_dir,
            bert_tokenizer=bert_tokenizer,
            clip_tokenizer=clip_tokenizer,
            clip_processor=clip_processor,
            max_samples=args.max_samples
        )
    
    # 创建测试数据集（可能使用不同的图像目录）
    test_dataset = MultiModalDataset(
        data_path=test_path,
        images_dir=test_images_dir,  # 使用测试图像目录
        bert_tokenizer=bert_tokenizer,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        max_samples=args.max_samples
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  
        pin_memory=True
    )
    
    # 创建融合模型
    print("创建融合模型...")
    fusion_model = create_fusion_model(
        text_model=text_model,
        image_model=image_model,
        fusion_dim=args.fusion_dim,
        num_classes=args.num_classes
    )
    
    # 训练模型
    if not args.eval_only:
        print("开始训练模型...")
        fusion_model, history = train_model(
            model=fusion_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            save_path=fusion_model_path,
            patience=args.patience
        )
        
        # 绘制训练历史
        plot_training_history(history)
    else:
        # 加载已有模型
        if os.path.exists(fusion_model_path):
            print(f"从 {fusion_model_path} 加载融合模型")
            checkpoint = torch.load(fusion_model_path, map_location=device)
            fusion_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            print(f"错误：评估模式下需要提供有效的模型路径，但 {fusion_model_path} 不存在")
            return
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(
        model=fusion_model,
        test_loader=test_loader,
        device=device
    )
    
    # 保存评估结果
    import json
    with open(os.path.join('fusionModels', 'evaluation_results.json'), 'w') as f:
        # 转换numpy数组为列表
        metrics_json = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'confusion_matrix': metrics['confusion_matrix'].tolist()
        }
        json.dump(metrics_json, f, indent=4)
    print(f"评估结果已保存到 {os.path.join('fusionModels', 'evaluation_results.json')}")

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='训练和评估多模态融合模型')
    
    # 数据参数
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data',
                        help='数据目录')
    parser.add_argument('--train_file', type=str, default='twitter_dataset/devset/posts.txt',
                        help='训练数据文件名')
    parser.add_argument('--val_file', type=str, default='twitter_dataset/testset/posts_groundtruth.txt',
                        help='验证数据文件名')
    parser.add_argument('--test_file', type=str, default='twitter_dataset/testset/posts_groundtruth.txt',
                        help='测试数据文件名')
    parser.add_argument('--images_dir', type=str, default='twitter_dataset/devset/images',
                        help='图像目录名')
    parser.add_argument('--test_images_dir', type=str, default='twitter_dataset/testset/images',
                        help='测试图像目录名')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数，用于调试')
    parser.add_argument('--train_val_split', type=float, default=0.8,
                        help='当使用相同数据文件时，训练集的比例')
    
    # 模型参数
    parser.add_argument('--text_model_path', type=str, default='saved_models/text_model.pth',
                        help='文本模型路径')
    parser.add_argument('--image_model_path', type=str, default='saved_models/image_model.pth',
                        help='图像模型路径')
    parser.add_argument('--fusion_model_path', type=str, default='saved_models/fusion_model.pth',
                        help='融合模型保存路径')
    parser.add_argument('--fusion_dim', type=int, default=512,
                        help='融合特征维度')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='分类类别数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='梯度累积步数')
    parser.add_argument('--patience', type=int, default=3,
                        help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='数据加载器工作进程数')
    
    # 其他参数
    parser.add_argument('--eval_only', action='store_true',
                        help='仅评估模型，不训练')
    parser.add_argument('--small_dataset', action='store_true',
                        help='使用小数据集进行测试')
    parser.add_argument('--model_cache_dir', type=str, default='/root/autodl-tmp/model_cache_new',
                        help='模型缓存目录')
    
    args = parser.parse_args()
    
    # 使用小数据集进行测试
    if args.small_dataset:
        args.train_file = 'small_train.txt'
        args.val_file = 'small_val.txt'
        args.test_file = 'small_test.txt'
    
    main(args)
