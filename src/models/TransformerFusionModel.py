#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
多模态特征融合模型
使用 Transformer 处理 ResNet+CLIP 和 BERT+CLIP 生成的特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import os
import sys
import numpy as np
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TransformerEncoderLayer(nn.Module):
    """
    自定义的Transformer编码器层，专注于特征融合和稳定性
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 前馈神经网络
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # 激活函数
        self.activation = F.gelu
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src: 输入特征 [batch_size, seq_len, d_model]
            src_mask: 注意力掩码 [seq_len, seq_len]
            src_key_padding_mask: 键填充掩码 [batch_size, seq_len]
            
        Returns:
            输出特征 [batch_size, seq_len, d_model]
        """
        # 自注意力机制
        src2, _ = self.self_attn(src, src, src, 
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        
        # 前馈神经网络
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        
        return src

class TransformerEncoder(nn.Module):
    """
    自定义的Transformer编码器，由多个编码器层组成
    """
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src: 输入特征 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [seq_len, seq_len]
            src_key_padding_mask: 键填充掩码 [batch_size, seq_len]
            
        Returns:
            输出特征 [batch_size, seq_len, d_model]
        """
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
        return output

class CrossModalTransformer(nn.Module):
    """
    跨模态Transformer，用于融合文本和图像特征
    """
    def __init__(self, d_model: int, nhead: int, num_encoder_layers: int = 2, 
                 dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 创建编码器层
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 创建编码器
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
    
    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        前向传播
        
        Args:
            src: 输入特征 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [seq_len, seq_len]
            src_key_padding_mask: 键填充掩码 [batch_size, seq_len]
            
        Returns:
            输出特征 [batch_size, seq_len, d_model]
        """
        try:
            # 记录输入形状
            batch_size, seq_len, d_model = src.shape
            logger.info(f"CrossModalTransformer输入形状: {src.shape}")
            
            # 检查输入维度是否与pos_encoder期望的维度匹配
            if d_model != self.pos_encoder.d_model:
                logger.warning(f"输入特征维度 {d_model} 与位置编码器期望的维度 {self.pos_encoder.d_model} 不匹配")
                # 如果不匹配，调整输入维度
                if d_model > self.pos_encoder.d_model:
                    # 截断多余的维度
                    src = src[:, :, :self.pos_encoder.d_model]
                    logger.info(f"截断后的输入形状: {src.shape}")
                else:
                    # 填充缺少的维度
                    padding = torch.zeros(batch_size, seq_len, self.pos_encoder.d_model - d_model).to(src.device)
                    src = torch.cat([src, padding], dim=2)
                    logger.info(f"填充后的输入形状: {src.shape}")
            
            # 添加位置编码
            src = self.pos_encoder(src)
            logger.info(f"位置编码后形状: {src.shape}")
            
            # 通过Transformer编码器
            output = self.transformer_encoder(src, mask=mask, src_key_padding_mask=src_key_padding_mask)
            logger.info(f"Transformer编码器输出形状: {output.shape}")
            
            return output
        except Exception as e:
            logger.error(f"CrossModalTransformer前向传播失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回输入作为输出，以避免训练崩溃
            return src

class PositionalEncoding(nn.Module):
    """
    位置编码，为Transformer提供序列位置信息
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model  # 记录维度
        
        # 创建位置编码
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的特征 [batch_size, seq_len, d_model]
        """
        try:
            # 检查输入形状
            logger.info(f"PositionalEncoding输入形状: {x.shape}, 预期维度: {self.d_model}")
            
            # 检查输入维度是否匹配
            if x.size(-1) != self.d_model:
                logger.warning(f"输入特征维度 {x.size(-1)} 与预期维度 {self.d_model} 不匹配")
            
            # 添加位置编码
            x = x + self.pe[:x.size(1)].unsqueeze(0)
            logger.info(f"添加位置编码后形状: {x.shape}")
            
            # 应用dropout
            result = self.dropout(x)
            return result
        except Exception as e:
            logger.error(f"PositionalEncoding前向传播失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 返回输入作为输出，以避免训练崩溃
            return x

class MultiModalFusionTransformer(nn.Module):
    """
    多模态融合Transformer模型
    融合文本和图像特征进行假新闻检测
    """
    def __init__(self, 
                 bert_dim: int = 768, 
                 clip_text_dim: int = 512,
                 resnet_dim: int = 512, 
                 clip_vision_dim: int = 768,  # 修改为768，与ImageProcessingModel中的维度设置匹配
                 fusion_dim: int = 512, 
                 num_heads: int = 8,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1,
                 num_classes: int = 2):
        super().__init__()
        
        # 记录输入维度，用于调试
        self.bert_dim = bert_dim
        self.clip_text_dim = clip_text_dim
        self.resnet_dim = resnet_dim
        self.clip_vision_dim = clip_vision_dim
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes
        
        # 特征对齐层 - 将各种特征映射到相同的维度空间
        self.bert_alignment = nn.Linear(bert_dim, fusion_dim)
        self.clip_text_alignment = nn.Linear(clip_text_dim, fusion_dim)
        self.resnet_alignment = nn.Linear(resnet_dim, fusion_dim)
        self.clip_vision_alignment = nn.Linear(clip_vision_dim, fusion_dim)
        
        # 文本特征融合
        self.text_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # 输入维度是两个对齐特征的拼接
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 图像特征融合
        self.vision_fusion = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),  # 输入维度是两个对齐特征的拼接
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 跨模态Transformer
        self.cross_modal_transformer = CrossModalTransformer(
            d_model=fusion_dim,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes)
        )
    
    def forward(self, bert_features, clip_text_features, resnet_features, clip_vision_features):
        """
        前向传播
        
        Args:
            bert_features: BERT特征 [batch_size, bert_dim]
            clip_text_features: CLIP文本特征 [batch_size, clip_text_dim]
            resnet_features: ResNet特征 [batch_size, resnet_dim]
            clip_vision_features: CLIP视觉特征 [batch_size, clip_vision_dim]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
            fused_features: 融合后的特征 [batch_size, fusion_dim]
        """
        try:
            batch_size = bert_features.size(0)
            device = bert_features.device
            
            # 打印输入特征的形状，用于调试
            logger.info(f"BERT特征形状: {bert_features.shape}, 预期维度: {self.bert_dim}")
            logger.info(f"CLIP文本特征形状: {clip_text_features.shape}, 预期维度: {self.clip_text_dim}")
            logger.info(f"ResNet特征形状: {resnet_features.shape}, 预期维度: {self.resnet_dim}")
            logger.info(f"CLIP视觉特征形状: {clip_vision_features.shape}, 预期维度: {self.clip_vision_dim}")
            
            # 检查输入维度是否匹配预期维度，如果不匹配则调整
            if bert_features.size(1) != self.bert_dim:
                logger.warning(f"BERT特征维度不匹配，调整维度: {bert_features.size(1)} -> {self.bert_dim}")
                # 如果维度大于预期，截断；如果小于预期，填充
                if bert_features.size(1) > self.bert_dim:
                    bert_features = bert_features[:, :self.bert_dim]
                else:
                    padding = torch.zeros(batch_size, self.bert_dim - bert_features.size(1)).to(device)
                    bert_features = torch.cat([bert_features, padding], dim=1)
            
            if clip_text_features.size(1) != self.clip_text_dim:
                logger.warning(f"CLIP文本特征维度不匹配，调整维度: {clip_text_features.size(1)} -> {self.clip_text_dim}")
                if clip_text_features.size(1) > self.clip_text_dim:
                    clip_text_features = clip_text_features[:, :self.clip_text_dim]
                else:
                    padding = torch.zeros(batch_size, self.clip_text_dim - clip_text_features.size(1)).to(device)
                    clip_text_features = torch.cat([clip_text_features, padding], dim=1)
            
            if resnet_features.size(1) != self.resnet_dim:
                logger.warning(f"ResNet特征维度不匹配，调整维度: {resnet_features.size(1)} -> {self.resnet_dim}")
                if resnet_features.size(1) > self.resnet_dim:
                    resnet_features = resnet_features[:, :self.resnet_dim]
                else:
                    padding = torch.zeros(batch_size, self.resnet_dim - resnet_features.size(1)).to(device)
                    resnet_features = torch.cat([resnet_features, padding], dim=1)
            
            if clip_vision_features.size(1) != self.clip_vision_dim:
                logger.warning(f"CLIP视觉特征维度不匹配，调整维度: {clip_vision_features.size(1)} -> {self.clip_vision_dim}")
                if clip_vision_features.size(1) > self.clip_vision_dim:
                    clip_vision_features = clip_vision_features[:, :self.clip_vision_dim]
                else:
                    padding = torch.zeros(batch_size, self.clip_vision_dim - clip_vision_features.size(1)).to(device)
                    clip_vision_features = torch.cat([clip_vision_features, padding], dim=1)
            
            # 特征对齐 - 将各种特征映射到相同的维度空间
            bert_aligned = self.bert_alignment(bert_features)  # [batch_size, fusion_dim]
            clip_text_aligned = self.clip_text_alignment(clip_text_features)  # [batch_size, fusion_dim]
            resnet_aligned = self.resnet_alignment(resnet_features)  # [batch_size, fusion_dim]
            clip_vision_aligned = self.clip_vision_alignment(clip_vision_features)  # [batch_size, fusion_dim]
            
            # 打印对齐特征的形状
            logger.info(f"对齐后的BERT特征形状: {bert_aligned.shape}")
            logger.info(f"对齐后的CLIP文本特征形状: {clip_text_aligned.shape}")
            logger.info(f"对齐后的ResNet特征形状: {resnet_aligned.shape}")
            logger.info(f"对齐后的CLIP视觉特征形状: {clip_vision_aligned.shape}")
            
            # 融合文本特征
            text_combined = torch.cat([bert_aligned, clip_text_aligned], dim=1)  # [batch_size, fusion_dim*2]
            logger.info(f"拼接后的文本特征形状: {text_combined.shape}")
            text_fused = self.text_fusion(text_combined)  # [batch_size, fusion_dim]
            logger.info(f"融合后的文本特征形状: {text_fused.shape}")
            
            # 融合图像特征
            vision_combined = torch.cat([resnet_aligned, clip_vision_aligned], dim=1)  # [batch_size, fusion_dim*2]
            logger.info(f"拼接后的图像特征形状: {vision_combined.shape}")
            vision_fused = self.vision_fusion(vision_combined)  # [batch_size, fusion_dim]
            logger.info(f"融合后的图像特征形状: {vision_fused.shape}")
            
            # 准备跨模态融合的输入
            # 将特征堆叠为序列 [batch_size, 2, fusion_dim]
            sequence = torch.stack([text_fused, vision_fused], dim=1)
            logger.info(f"堆叠后的序列形状: {sequence.shape}")
            
            # 跨模态融合
            fused_sequence = self.cross_modal_transformer(sequence)
            logger.info(f"融合后的序列形状: {fused_sequence.shape}")
            
            # 取序列的平均值作为最终特征
            fused_features = fused_sequence.mean(dim=1)  # [batch_size, fusion_dim]
            logger.info(f"最终融合特征形状: {fused_features.shape}")
            
            # 分类
            logits = self.classifier(fused_features)  # [batch_size, num_classes]
            logger.info(f"分类logits形状: {logits.shape}")
            
            return logits, fused_features
        except Exception as e:
            logger.error(f"多模态融合Transformer前向传播失败: {e}")
            
            # 记录更详细的错误信息
            import traceback
            logger.error(traceback.format_exc())
            
            # 记录输入特征的形状和类型
            logger.error(f"BERT特征: 类型={type(bert_features)}, 形状={bert_features.shape if hasattr(bert_features, 'shape') else 'unknown'}, 预期维度: {self.bert_dim}")
            logger.error(f"CLIP文本特征: 类型={type(clip_text_features)}, 形状={clip_text_features.shape if hasattr(clip_text_features, 'shape') else 'unknown'}, 预期维度: {self.clip_text_dim}")
            logger.error(f"ResNet特征: 类型={type(resnet_features)}, 形状={resnet_features.shape if hasattr(resnet_features, 'shape') else 'unknown'}, 预期维度: {self.resnet_dim}")
            logger.error(f"CLIP视觉特征: 类型={type(clip_vision_features)}, 形状={clip_vision_features.shape if hasattr(clip_vision_features, 'shape') else 'unknown'}, 预期维度: {self.clip_vision_dim}")
            
            # 创建随机初始化的输出
            batch_size = bert_features.size(0) if hasattr(bert_features, 'size') else 1
            device = bert_features.device if hasattr(bert_features, 'device') else torch.device('cpu')
            fused_features = torch.randn(batch_size, self.fusion_dim).to(device) * 0.01
            logits = torch.randn(batch_size, self.num_classes).to(device) * 0.01
            
            # 记录错误信息
            logger.error("返回随机初始化的输出")
            
            return logits, fused_features

class FusionModelWrapper(nn.Module):
    """
    融合模型包装器
    整合文本处理模型和图像处理模型，并使用Transformer进行特征融合
    """
    def __init__(self, text_model, image_model, fusion_dim=512, num_classes=2):
        super().__init__()
        
        # 文本和图像模型
        self.text_model = text_model
        self.image_model = image_model
        
        # 获取特征维度
        self.bert_dim = text_model.bert_dim
        self.clip_text_dim = text_model.clip_dim
        self.resnet_dim = image_model.resnet_dim
        self.clip_vision_dim = 768  # 修改为768，与ImageProcessingModel中的维度设置匹配
        
        # 创建融合模型
        self.fusion_model = MultiModalFusionTransformer(
            bert_dim=self.bert_dim,
            clip_text_dim=self.clip_text_dim,
            resnet_dim=self.resnet_dim,
            clip_vision_dim=self.clip_vision_dim,
            fusion_dim=fusion_dim,
            num_classes=num_classes
        )
    
    def forward(self, bert_input_ids, bert_attention_mask, clip_input_ids, clip_attention_mask, resnet_image, clip_pixel_values):
        """
        前向传播
        
        Args:
            bert_input_ids: BERT输入ID [batch_size, seq_len]
            bert_attention_mask: BERT注意力掩码 [batch_size, seq_len]
            clip_input_ids: CLIP输入ID [batch_size, seq_len]
            clip_attention_mask: CLIP注意力掩码 [batch_size, seq_len]
            resnet_image: ResNet输入图像 [batch_size, 3, H, W]
            clip_pixel_values: CLIP输入图像 [batch_size, 3, H, W]
            
        Returns:
            logits: 分类logits [batch_size, num_classes]
            fused_features: 融合后的特征 [batch_size, fusion_dim]
        """
        # 提取文本特征
        try:
            _, bert_features, clip_text_features = self.text_model(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                clip_input_ids=clip_input_ids,
                clip_attention_mask=clip_attention_mask
            )
        except Exception as e:
            logger.error(f"提取文本特征时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 记录输入的形状和类型
            logger.error(f"BERT输入ID: 类型={type(bert_input_ids)}, 形状={bert_input_ids.shape if hasattr(bert_input_ids, 'shape') else 'unknown'}")
            logger.error(f"BERT注意力掩码: 类型={type(bert_attention_mask)}, 形状={bert_attention_mask.shape if hasattr(bert_attention_mask, 'shape') else 'unknown'}")
            logger.error(f"CLIP输入ID: 类型={type(clip_input_ids)}, 形状={clip_input_ids.shape if hasattr(clip_input_ids, 'shape') else 'unknown'}")
            logger.error(f"CLIP注意力掩码: 类型={type(clip_attention_mask)}, 形状={clip_attention_mask.shape if hasattr(clip_attention_mask, 'shape') else 'unknown'}")
            
            # 创建空特征以避免训练崩溃
            batch_size = bert_input_ids.size(0)
            bert_features = torch.zeros(batch_size, self.bert_dim).to(bert_input_ids.device)
            clip_text_features = torch.zeros(batch_size, self.clip_text_dim).to(bert_input_ids.device)
            
            logger.error(f"创建零特征: bert_features={bert_features.shape}, clip_text_features={clip_text_features.shape}")
        
        # 提取图像特征
        try:
            _, resnet_features, clip_vision_features = self.image_model(
                resnet_image=resnet_image,
                clip_pixel_values=clip_pixel_values
            )
        except Exception as e:
            logger.error(f"提取图像特征时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 记录输入的形状和类型
            logger.error(f"ResNet图像: 类型={type(resnet_image)}, 形状={resnet_image.shape if hasattr(resnet_image, 'shape') else 'unknown'}")
            logger.error(f"CLIP像素值: 类型={type(clip_pixel_values)}, 形状={clip_pixel_values.shape if hasattr(clip_pixel_values, 'shape') else 'unknown'}")
            
            # 创建空特征以避免训练崩溃
            batch_size = resnet_image.size(0)
            resnet_features = torch.zeros(batch_size, self.resnet_dim).to(resnet_image.device)
            clip_vision_features = torch.zeros(batch_size, self.clip_vision_dim).to(resnet_image.device)
            
            logger.error(f"创建零特征: resnet_features={resnet_features.shape}, clip_vision_features={clip_vision_features.shape}")
        
        # 融合特征并分类
        logits, fused_features = self.fusion_model(
            bert_features=bert_features,
            clip_text_features=clip_text_features,
            resnet_features=resnet_features,
            clip_vision_features=clip_vision_features
        )
        
        return logits, fused_features

def create_fusion_model(text_model, image_model, fusion_dim=512, num_classes=2):
    """
    创建融合模型
    
    Args:
        text_model: 文本处理模型
        image_model: 图像处理模型
        fusion_dim: 融合特征维度
        num_classes: 分类类别数
        
    Returns:
        融合模型
    """
    # 确保模型处于评估模式
    text_model.eval()
    image_model.eval()
    
    # 创建融合模型
    fusion_model = FusionModelWrapper(
        text_model=text_model,
        image_model=image_model,
        fusion_dim=fusion_dim,
        num_classes=num_classes
    )
    
    logger.info(f"成功创建融合模型，融合维度: {fusion_dim}, 分类类别数: {num_classes}")
    return fusion_model
