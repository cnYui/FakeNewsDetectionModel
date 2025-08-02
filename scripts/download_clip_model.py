#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载完整的CLIP模型（包括文本和视觉部分）
"""

import os
import argparse
import logging
from transformers import CLIPModel, CLIPProcessor

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_clip_model(model_name, save_dir, use_auth_token=None):
    """
    下载CLIP模型并保存到本地
    
    Args:
        model_name: 模型名称，例如 'openai/clip-vit-base-patch32'
        save_dir: 保存目录
        use_auth_token: Hugging Face认证令牌（如果需要）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 模型保存路径
    model_save_path = os.path.join(save_dir, model_name.split('/')[-1])
    
    # 创建模型保存目录
    os.makedirs(model_save_path, exist_ok=True)
    
    logger.info(f"开始下载CLIP模型: {model_name}")
    
    try:
        # 下载CLIP模型
        model = CLIPModel.from_pretrained(model_name, use_auth_token=use_auth_token)
        
        # 保存模型
        model.save_pretrained(model_save_path)
        logger.info(f"CLIP模型已保存到: {model_save_path}")
        
        # 下载CLIP处理器
        processor = CLIPProcessor.from_pretrained(model_name, use_auth_token=use_auth_token)
        
        # 保存处理器
        processor.save_pretrained(model_save_path)
        logger.info(f"CLIP处理器已保存到: {model_save_path}")
        
        # 验证模型是否包含文本和视觉部分
        if hasattr(model, 'text_model') and hasattr(model, 'vision_model'):
            logger.info("验证成功: 模型包含文本模型和视觉模型")
        else:
            logger.warning("警告: 模型可能不包含完整的文本和视觉部分")
            if hasattr(model, 'text_model'):
                logger.info("模型包含文本模型")
            else:
                logger.warning("模型不包含文本模型")
            
            if hasattr(model, 'vision_model'):
                logger.info("模型包含视觉模型")
            else:
                logger.warning("模型不包含视觉模型")
        
        return True
    except Exception as e:
        logger.error(f"下载CLIP模型失败: {e}")
        return False

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="下载CLIP模型")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", 
                        help="模型名称，例如 'openai/clip-vit-base-patch32'")
    parser.add_argument("--save_dir", type=str, default="/Users/wujianxiang/Documents/GitHub/models/model_cache", 
                        help="保存目录")
    parser.add_argument("--use_auth_token", type=str, default=None, 
                        help="Hugging Face认证令牌（如果需要）")
    
    args = parser.parse_args()
    
    # 下载模型
    success = download_clip_model(
        model_name=args.model_name,
        save_dir=args.save_dir,
        use_auth_token=args.use_auth_token
    )
    
    if success:
        logger.info("CLIP模型下载完成")
    else:
        logger.error("CLIP模型下载失败")

if __name__ == "__main__":
    main()
