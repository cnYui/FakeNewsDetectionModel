#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集对比分析脚本
对比当前使用的清洗数据集和融合数据集的差异
"""

import os
import json
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_csv_dataset(data_dir):
    """加载CSV格式的数据集"""
    train_df = pd.read_csv(os.path.join(data_dir, 'train_cleaned.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val_cleaned.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test_cleaned.csv'))
    
    return {
        'train': train_df,
        'val': val_df,
        'test': test_df
    }

def load_json_dataset(data_dir):
    """加载JSON格式的融合数据集"""
    with open(os.path.join(data_dir, 'merged_train.json'), 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    with open(os.path.join(data_dir, 'merged_val.json'), 'r', encoding='utf-8') as f:
        val_data = json.load(f)
    with open(os.path.join(data_dir, 'merged_test.json'), 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }

def analyze_csv_dataset(datasets):
    """分析CSV数据集"""
    print("=== CSV数据集分析（当前使用的清洗数据集）===")
    
    total_samples = 0
    for split, df in datasets.items():
        print(f"\n{split.upper()} 集合:")
        print(f"  样本数量: {len(df)}")
        print(f"  标签分布: {dict(df['label'].value_counts())}")
        print(f"  文本长度统计:")
        text_lengths = df['text'].str.len()
        print(f"    平均长度: {text_lengths.mean():.2f}")
        print(f"    最大长度: {text_lengths.max()}")
        print(f"    最小长度: {text_lengths.min()}")
        total_samples += len(df)
    
    print(f"\n总样本数: {total_samples}")
    return total_samples

def analyze_json_dataset(datasets):
    """分析JSON数据集"""
    print("\n=== JSON数据集分析（融合数据集）===")
    
    total_samples = 0
    source_stats = Counter()
    
    for split, data in datasets.items():
        print(f"\n{split.upper()} 集合:")
        print(f"  样本数量: {len(data)}")
        
        # 标签分布
        labels = [item['label'] for item in data]
        label_dist = dict(Counter(labels))
        print(f"  标签分布: {label_dist}")
        
        # 数据来源统计
        sources = [item.get('source', 'unknown') for item in data]
        source_dist = dict(Counter(sources))
        print(f"  数据来源分布: {source_dist}")
        
        # 更新总体来源统计
        source_stats.update(sources)
        
        # 文本长度统计
        text_lengths = [len(item['text']) for item in data]
        print(f"  文本长度统计:")
        print(f"    平均长度: {sum(text_lengths)/len(text_lengths):.2f}")
        print(f"    最大长度: {max(text_lengths)}")
        print(f"    最小长度: {min(text_lengths)}")
        
        # 检查是否有domain字段
        has_domain = any('domain' in item for item in data)
        if has_domain:
            domains = [item.get('domain', 0) for item in data]
            domain_dist = dict(Counter(domains))
            print(f"  域分布: {domain_dist}")
        
        total_samples += len(data)
    
    print(f"\n总样本数: {total_samples}")
    print(f"\n整体数据来源统计: {dict(source_stats)}")
    return total_samples, dict(source_stats)

def compare_datasets():
    """对比两个数据集"""
    print("数据集对比分析")
    print("=" * 50)
    
    # 加载CSV数据集（当前使用的清洗数据集）
    csv_dir = '/root/autodl-tmp/data/cleaned'
    if os.path.exists(csv_dir):
        csv_datasets = load_csv_dataset(csv_dir)
        csv_total = analyze_csv_dataset(csv_datasets)
    else:
        print(f"CSV数据集目录不存在: {csv_dir}")
        csv_total = 0
    
    # 加载JSON数据集（融合数据集）
    json_dirs = [
        '/root/autodl-tmp/data/dataset/merged',
        '/root/autodl-tmp/dataset/merged'
    ]
    
    json_total = 0
    source_stats = {}
    
    for json_dir in json_dirs:
        if os.path.exists(json_dir):
            print(f"\n找到融合数据集: {json_dir}")
            json_datasets = load_json_dataset(json_dir)
            json_total, source_stats = analyze_json_dataset(json_datasets)
            break
    else:
        print("\n未找到融合数据集")
    
    # 对比总结
    print("\n=== 数据集对比总结 ===")
    print(f"当前使用的清洗数据集总样本数: {csv_total}")
    print(f"融合数据集总样本数: {json_total}")
    print(f"样本数量差异: {json_total - csv_total}")
    
    if json_total > 0:
        print(f"\n融合数据集的优势:")
        print(f"1. 数据量更大: {json_total} vs {csv_total} (+{json_total - csv_total} 样本)")
        print(f"2. 多数据源融合: {list(source_stats.keys())}")
        print(f"3. 包含域信息: 支持跨域检测")
        print(f"4. 数据格式统一: JSON格式便于扩展")
        
        print(f"\n建议:")
        print(f"1. 考虑使用融合数据集进行训练，可能获得更好的性能")
        print(f"2. 融合数据集包含更多样化的数据源，有助于模型泛化")
        print(f"3. 可以利用域信息进行跨域假新闻检测研究")

def check_data_format_compatibility():
    """检查数据格式兼容性"""
    print("\n=== 数据格式兼容性检查 ===")
    
    # 检查JSON数据集的字段
    json_dirs = [
        '/root/autodl-tmp/data/dataset/merged',
        '/root/autodl-tmp/dataset/merged'
    ]
    
    for json_dir in json_dirs:
        if os.path.exists(json_dir):
            with open(os.path.join(json_dir, 'merged_train.json'), 'r', encoding='utf-8') as f:
                sample_data = json.load(f)[:5]  # 只看前5个样本
            
            print(f"\n融合数据集样本格式 ({json_dir}):")
            for i, item in enumerate(sample_data):
                print(f"样本 {i+1}: {list(item.keys())}")
                if i == 0:
                    print(f"  示例数据: {item}")
            
            # 检查与当前训练脚本的兼容性
            required_fields = ['text', 'image', 'label']
            sample_fields = set(sample_data[0].keys())
            
            print(f"\n兼容性检查:")
            print(f"当前训练脚本需要的字段: {required_fields}")
            print(f"融合数据集包含的字段: {list(sample_fields)}")
            
            missing_fields = set(required_fields) - sample_fields
            extra_fields = sample_fields - set(required_fields)
            
            if not missing_fields:
                print("✅ 融合数据集与当前训练脚本完全兼容")
            else:
                print(f"❌ 缺少必需字段: {missing_fields}")
            
            if extra_fields:
                print(f"ℹ️  额外字段: {extra_fields} (可用于扩展功能)")
            
            break

if __name__ == '__main__':
    compare_datasets()
    check_data_format_compatibility()