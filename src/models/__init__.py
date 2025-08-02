# -*- coding: utf-8 -*-
"""
融合模型包
包含多模态特征融合模型的定义和训练代码
"""

# 导出模块
from .TransformerFusionModel import (
    MultiModalFusionTransformer,
    CrossModalTransformer,
    TransformerEncoder,
    TransformerEncoderLayer,
    PositionalEncoding,
    FusionModelWrapper,
    create_fusion_model
)
