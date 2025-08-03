#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练好的多模态假新闻检测模型
"""

import torch
import pandas as pd
import os
import sys
import logging
from datetime import datetime

# 添加项目路径
sys.path.append('/root/models')
from src.models.MultiModalFakeNewsDetector import FakeNewsDataPreprocessor
from train_multimodal_model import MultiModalFusionModel

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model(model_path, device):
    """
    加载训练好的模型
    """
    logger.info(f"正在加载模型: {model_path}")
    
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取模型参数
    args = checkpoint['args']
    
    # 创建模型
    model = MultiModalFusionModel(
        text_dim=768,  # BERT特征维度
        image_dim=768,  # CLIP特征维度
        hidden_dim=args['hidden_dim'],
        num_classes=2,
        dropout=args['dropout']
    )
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info("模型加载完成")
    return model, args

def test_single_prediction(model, preprocessor, text, image_path, device):
    """
    测试单个样本的预测
    """
    try:
        # 编码文本和图像
        text_encoded = preprocessor.encode_text(text)
        image_encoded = preprocessor.encode_image(image_path)
        
        if text_encoded is None or image_encoded is None:
            return None
        
        # 提取特征
        text_features = preprocessor.get_text_features(text_encoded)
        image_features = preprocessor.get_image_features(image_encoded)
        
        # 确保特征在正确的设备上
        text_features = text_features.to(device)
        image_features = image_features.to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(text_features, image_features)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy().tolist(),
            'prediction': 'Real' if predicted_class == 0 else 'Fake'
        }
    
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return None

def test_model_on_dataset(model, preprocessor, test_df, device, max_samples=10):
    """
    在测试数据集上测试模型
    """
    logger.info(f"开始测试模型，最多测试 {max_samples} 个样本")
    
    correct_predictions = 0
    total_predictions = 0
    results = []
    
    for idx, row in test_df.head(max_samples).iterrows():
        text = row['text']
        # 处理路径格式，移除多余的前缀
        path = row['path']
        if path.startswith('./data/images/'):
            path = path.replace('./data/images/', '')
        image_path = os.path.join('/root/autodl-tmp/data/images', path)
        true_label = int(row['label'])
        
        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            logger.warning(f"图像文件不存在: {image_path}")
            continue
        
        # 进行预测
        prediction_result = test_single_prediction(model, preprocessor, text, image_path, device)
        
        if prediction_result is not None:
            predicted_class = prediction_result['predicted_class']
            confidence = prediction_result['confidence']
            
            # 统计准确率
            if predicted_class == true_label:
                correct_predictions += 1
            total_predictions += 1
            
            # 记录结果
            result = {
                'index': idx,
                'text': text[:100] + '...' if len(text) > 100 else text,
                'image_path': row['path'],
                'true_label': true_label,
                'predicted_class': predicted_class,
                'prediction': prediction_result['prediction'],
                'confidence': confidence,
                'correct': predicted_class == true_label
            }
            results.append(result)
            
            logger.info(f"样本 {idx}: 真实={true_label}, 预测={predicted_class}, 置信度={confidence:.4f}, 正确={predicted_class == true_label}")
    
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    logger.info(f"测试完成: {correct_predictions}/{total_predictions} 正确, 准确率: {accuracy:.4f}")
    
    return results, accuracy

def main():
    logger.info("开始测试训练好的多模态假新闻检测模型")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 模型路径
    model_path = '/root/models/checkpoints/multimodal_model.pth'
    
    try:
        # 加载模型
        model, args = load_trained_model(model_path, device)
        
        # 初始化数据预处理器
        logger.info("初始化数据预处理器...")
        preprocessor = FakeNewsDataPreprocessor()
        
        # 测试模型
        test_df = preprocessor.test_df
        results, accuracy = test_model_on_dataset(model, preprocessor, test_df, device, max_samples=10)
        
        # 保存测试结果
        results_df = pd.DataFrame(results)
        results_path = '/root/models/checkpoints/test_results.csv'
        results_df.to_csv(results_path, index=False, encoding='utf-8')
        logger.info(f"测试结果已保存到: {results_path}")
        
        # 显示总结
        logger.info("=" * 50)
        logger.info("测试总结:")
        logger.info(f"模型路径: {model_path}")
        logger.info(f"测试样本数: {len(results)}")
        logger.info(f"准确率: {accuracy:.4f}")
        logger.info(f"模型参数: {args}")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()