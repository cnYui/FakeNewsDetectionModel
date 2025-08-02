#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
用于测试CLIP模型加载的脚本
"""

import os
import torch
from transformers import CLIPTokenizer, CLIPProcessor, CLIPImageProcessor

# CLIP模型路径
clip_model_path = '/root/models/model_cache/clip-vit-base-patch32(new)'
print(f"从本地文件系统加载CLIP模型: {clip_model_path}")

# 准备CLIP分词器所需的文件路径
vocab_file = os.path.join(clip_model_path, 'vocab.json')
merges_file = os.path.join(clip_model_path, 'merges.txt')

# 检查文件是否存在
if os.path.exists(vocab_file) and os.path.exists(merges_file):
    print(f"找到CLIP分词器所需的文件: vocab.json 和 merges.txt")
    
    # 使用transformers的CLIPTokenizer
    try:
        # 直接使用文件路径创建分词器
        clip_tokenizer = CLIPTokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            errors='replace',
            model_max_length=77,
            bos_token="<|startoftext|>",
            eos_token="
