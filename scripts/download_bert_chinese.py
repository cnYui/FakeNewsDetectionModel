#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
下载 google-bert/bert-base-chinese 模型到指定文件夹，支持断点续传和错误重试
"""

import os
import argparse
import time
import logging
from pathlib import Path
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def download_bert_chinese(output_dir, retry_count=5, retry_delay=5):
    """
    下载 google-bert/bert-base-chinese 模型到指定目录，支持断点续传和错误重试
    
    Args:
        output_dir: 输出目录路径
        retry_count: 重试次数
        retry_delay: 重试延迟（秒）
    """
    # 确保输出目录存在
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"开始下载 google-bert/bert-base-chinese 模型到 {output_dir}")
    
    # 模型名称
    model_name = "google-bert/bert-base-chinese"
    
    # 创建模型输出目录
    model_output_dir = output_path / "bert-base-chinese"
    model_output_dir.mkdir(exist_ok=True)
    
    # 尝试下载模型，支持重试
    for attempt in range(retry_count):
        try:
            logger.info(f"下载尝试 {attempt + 1}/{retry_count}")
            
            # 使用 transformers 直接下载模型
            try:
                from transformers import AutoTokenizer, AutoModelForMaskedLM
                
                logger.info("正在下载分词器...")
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    local_files_only=False,
                    cache_dir=str(output_path),
                    resume_download=True,
                    force_download=False
                )
                
                logger.info("正在下载模型...")
                model = AutoModelForMaskedLM.from_pretrained(
                    model_name,
                    local_files_only=False,
                    cache_dir=str(output_path),
                    resume_download=True,
                    force_download=False
                )
                
                logger.info(f"保存模型到 {model_output_dir}")
                model.save_pretrained(model_output_dir)
                tokenizer.save_pretrained(model_output_dir)
                
                logger.info("模型下载和保存完成！")
                
            except Exception as e:
                logger.error(f"使用transformers下载失败: {e}")
                raise
            
            # 验证下载的模型
            try:
                logger.info("\n验证模型...")
                # 尝试加载模型和分词器
                tokenizer = AutoTokenizer.from_pretrained(str(model_output_dir))
                model = AutoModelForMaskedLM.from_pretrained(str(model_output_dir))
                
                # 测试模型
                logger.info("测试模型功能:")
                text = "今天[MASK]情很好"
                logger.info(f"输入: {text}")
                
                inputs = tokenizer(text, return_tensors="pt")
                outputs = model(**inputs)
                
                mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
                predicted_token_id = outputs.logits[0, mask_token_index].argmax(axis=-1)
                predicted_token = tokenizer.decode(predicted_token_id)
                
                logger.info(f"预测结果: 今天{predicted_token}情很好")
                logger.info("模型验证成功！")
                
                return True
            except Exception as e:
                logger.error(f"模型验证失败: {e}")
                if attempt < retry_count - 1:
                    logger.info(f"将在 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
                
        except Exception as e:
            logger.error(f"下载失败: {e}")
            if attempt < retry_count - 1:
                logger.info(f"将在 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            else:
                logger.error(f"已达到最大重试次数 ({retry_count})，下载失败")
                return False
    
    return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载 BERT 中文模型")
    parser.add_argument("--output_dir", type=str, default="./model_cache", 
                        help="模型保存目录")
    parser.add_argument("--retry", type=int, default=5, 
                        help="下载失败时的重试次数")
    parser.add_argument("--delay", type=int, default=5, 
                        help="重试之间的延迟时间（秒）")
    args = parser.parse_args()
    
    success = download_bert_chinese(args.output_dir, args.retry, args.delay)
    
    if success:
        logger.info("脚本执行成功")
    else:
        logger.error("脚本执行失败")
        sys.exit(1)
