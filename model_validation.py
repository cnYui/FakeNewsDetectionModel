#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型验证脚本
用于验证多模态假新闻检测系统的各个组件
"""

import os
import sys
import torch
import pandas as pd
from PIL import Image
import numpy as np

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_dir)

def test_environment():
    """
    测试环境配置
    """
    print("=" * 60)
    print("环境配置验证")
    print("=" * 60)
    
    # Python版本
    print(f"Python版本: {sys.version}")
    
    # PyTorch版本和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    
    print("✓ 环境配置验证完成")

def test_data_loading():
    """
    测试数据加载
    """
    print("\n" + "=" * 60)
    print("数据加载验证")
    print("=" * 60)
    
    data_dir = '/root/autodl-tmp/data'
    
    # 测试CSV文件加载
    try:
        train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
        test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
        
        print(f"✓ 训练集: {len(train_df)} 条记录")
        print(f"✓ 验证集: {len(val_df)} 条记录")
        print(f"✓ 测试集: {len(test_df)} 条记录")
        
        # 检查数据格式
        required_columns = ['path', 'text', 'label']
        for df_name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"✗ {df_name}缺少列: {missing_cols}")
            else:
                print(f"✓ {df_name}数据格式正确")
        
        # 测试图像加载
        images_dir = os.path.join(data_dir, 'images')
        sample_path = train_df['path'].iloc[0]
        if sample_path.startswith('./data/images/'):
            image_name = sample_path.replace('./data/images/', '')
        else:
            image_name = os.path.basename(sample_path)
        
        image_path = os.path.join(images_dir, image_name)
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                print(f"✓ 图像加载成功: {img.size}")
            except Exception as e:
                print(f"✗ 图像加载失败: {e}")
        else:
            print(f"✗ 图像文件不存在: {image_path}")
            
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        return False
    
    print("✓ 数据加载验证完成")
    return True

def test_model_loading():
    """
    测试预训练模型加载
    """
    print("\n" + "=" * 60)
    print("预训练模型验证")
    print("=" * 60)
    
    try:
        from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor
        
        model_cache_dir = '/root/autodl-tmp/model_cache_new'
        
        # 测试BERT模型
        bert_path = os.path.join(model_cache_dir, 'bert-base-chinese')
        if os.path.exists(bert_path):
            bert_model = BertModel.from_pretrained(bert_path)
            bert_tokenizer = BertTokenizer.from_pretrained(bert_path)
            print(f"✓ BERT模型加载成功: hidden_size={bert_model.config.hidden_size}")
            
            # 测试BERT编码
            test_text = "这是一个测试文本"
            inputs = bert_tokenizer(test_text, return_tensors='pt', padding=True, truncation=True)
            with torch.no_grad():
                outputs = bert_model(**inputs)
            print(f"✓ BERT编码测试成功: {outputs.last_hidden_state.shape}")
        else:
            print(f"✗ BERT模型路径不存在: {bert_path}")
        
        # 测试CLIP模型
        clip_path = os.path.join(model_cache_dir, 'clip-vit-base-patch32')
        if os.path.exists(clip_path):
            clip_model = CLIPModel.from_pretrained(clip_path)
            clip_processor = CLIPProcessor.from_pretrained(clip_path)
            print(f"✓ CLIP模型加载成功: vision_hidden_size={clip_model.config.vision_config.hidden_size}")
            
            # 测试CLIP编码
            test_image = Image.new('RGB', (224, 224), color='red')
            inputs = clip_processor(images=test_image, return_tensors='pt')
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
            print(f"✓ CLIP编码测试成功: {image_features.shape}")
        else:
            print(f"✗ CLIP模型路径不存在: {clip_path}")
            
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return False
    
    print("✓ 预训练模型验证完成")
    return True

def test_model_components():
    """
    测试模型组件
    """
    print("\n" + "=" * 60)
    print("模型组件验证")
    print("=" * 60)
    
    try:
        # 测试导入
        from src.models.MultiModalFakeNewsDetector import FakeNewsDataPreprocessor
        from src.models.TransformerFusionModel import create_fusion_model
        
        print("✓ 模型组件导入成功")
        
        # 测试FakeNewsDataPreprocessor初始化
        preprocessor = FakeNewsDataPreprocessor()
        print("✓ FakeNewsDataPreprocessor初始化成功")
        
        # 测试融合模型创建 - 需要先创建文本和图像模型
        from src.models.text_models.TextProcessingModel import TextProcessingModel
        from src.models.image_models.ImageProcessingModel import ImageProcessingModel
        
        # 创建简单的文本和图像模型用于测试
        text_model = TextProcessingModel(num_classes=2)
        image_model = ImageProcessingModel(num_classes=2)
        
        fusion_model = create_fusion_model(
            text_model=text_model,
            image_model=image_model,
            fusion_dim=512,
            num_classes=2
        )
        print(f"✓ 融合模型创建成功")
        
        # 测试模型前向传播
        batch_size = 2
        # 创建模拟输入数据
        bert_input_ids = torch.randint(0, 1000, (batch_size, 128))
        bert_attention_mask = torch.ones(batch_size, 128)
        clip_input_ids = torch.randint(0, 1000, (batch_size, 77))
        clip_attention_mask = torch.ones(batch_size, 77)
        resnet_image = torch.randn(batch_size, 3, 224, 224)
        clip_pixel_values = torch.randn(batch_size, 3, 224, 224)
        
        with torch.no_grad():
            outputs = fusion_model(bert_input_ids, bert_attention_mask, clip_input_ids, clip_attention_mask, resnet_image, clip_pixel_values)
        print(f"✓ 模型前向传播测试成功: {outputs[0].shape}")
        
    except Exception as e:
        print(f"✗ 模型组件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("✓ 模型组件验证完成")
    return True

def main():
    """
    主验证函数
    """
    print("多模态假新闻检测系统验证")
    print("=" * 80)
    
    # 运行所有测试
    tests = [
        ("环境配置", test_environment),
        ("数据加载", test_data_loading),
        ("预训练模型", test_model_loading),
        ("模型组件", test_model_components)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result if result is not None else True
        except Exception as e:
            print(f"\n✗ {test_name}测试异常: {e}")
            results[test_name] = False
    
    # 输出总结
    print("\n" + "=" * 80)
    print("验证结果总结")
    print("=" * 80)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 所有验证测试通过！系统准备就绪。")
        print("\n下一步可以开始模型训练:")
        print("cd /root/models/src/models")
        print("python train_fusion_model.py --data_dir /root/autodl-tmp/data --model_cache_dir /root/autodl-tmp/model_cache_new")
    else:
        print("❌ 部分验证测试失败，请检查相关配置。")
    print("=" * 80)

if __name__ == '__main__':
    main()