#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集完整性分析脚本
用于检查多模态假新闻检测数据集的完整性和统计信息
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset():
    """
    分析数据集的完整性和统计信息
    """
    data_dir = '/root/autodl-tmp/data'
    images_dir = os.path.join(data_dir, 'images')
    
    print("=" * 60)
    print("多模态假新闻检测数据集分析报告")
    print("=" * 60)
    
    # 1. 检查文件存在性
    print("\n1. 文件存在性检查:")
    files_to_check = ['train.csv', 'val.csv', 'test.csv']
    for file in files_to_check:
        file_path = os.path.join(data_dir, file)
        exists = os.path.exists(file_path)
        size = os.path.getsize(file_path) / (1024*1024) if exists else 0
        print(f"  {file}: {'✓' if exists else '✗'} ({size:.2f} MB)")
    
    images_exist = os.path.exists(images_dir)
    image_count = len(os.listdir(images_dir)) if images_exist else 0
    print(f"  images目录: {'✓' if images_exist else '✗'} ({image_count} 个文件)")
    
    # 2. 数据集基本信息
    print("\n2. 数据集基本信息:")
    datasets = {}
    total_samples = 0
    
    for file in files_to_check:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            datasets[file.replace('.csv', '')] = df
            print(f"  {file}: {len(df)} 条记录")
            total_samples += len(df)
    
    print(f"  总计: {total_samples} 条记录")
    
    # 3. 标签分布分析
    print("\n3. 标签分布分析:")
    all_labels = []
    for name, df in datasets.items():
        labels = df['label'].value_counts()
        print(f"  {name}:")
        for label, count in labels.items():
            label_name = '假新闻' if label == 1 else '真新闻'
            percentage = count / len(df) * 100
            print(f"    {label_name}(标签{label}): {count} 条 ({percentage:.1f}%)")
            all_labels.extend(df['label'].tolist())
    
    # 总体标签分布
    total_labels = Counter(all_labels)
    print(f"  总体分布:")
    for label, count in total_labels.items():
        label_name = '假新闻' if label == 1 else '真新闻'
        percentage = count / len(all_labels) * 100
        print(f"    {label_name}(标签{label}): {count} 条 ({percentage:.1f}%)")
    
    # 4. 文本长度分析
    print("\n4. 文本长度分析:")
    for name, df in datasets.items():
        text_lengths = df['text'].str.len()
        print(f"  {name}:")
        print(f"    平均长度: {text_lengths.mean():.1f} 字符")
        print(f"    最短: {text_lengths.min()} 字符")
        print(f"    最长: {text_lengths.max()} 字符")
        print(f"    中位数: {text_lengths.median():.1f} 字符")
    
    # 5. 图像文件完整性检查
    print("\n5. 图像文件完整性检查:")
    missing_images = []
    corrupted_images = []
    sample_sizes = []
    
    # 检查前100个样本的图像文件
    print("  检查前100个样本的图像文件...")
    train_df = datasets.get('train', pd.DataFrame())
    if not train_df.empty:
        sample_paths = train_df['path'].head(100)
        for i, path in enumerate(sample_paths):
            # 处理路径格式
            if path.startswith('./data/images/'):
                image_name = path.replace('./data/images/', '')
            else:
                image_name = os.path.basename(path)
            
            image_path = os.path.join(images_dir, image_name)
            
            if not os.path.exists(image_path):
                missing_images.append(image_name)
            else:
                try:
                    with Image.open(image_path) as img:
                        sample_sizes.append(img.size)
                except Exception as e:
                    corrupted_images.append(image_name)
    
    print(f"    缺失图像: {len(missing_images)} 个")
    print(f"    损坏图像: {len(corrupted_images)} 个")
    print(f"    正常图像: {len(sample_sizes)} 个")
    
    if sample_sizes:
        widths = [size[0] for size in sample_sizes]
        heights = [size[1] for size in sample_sizes]
        print(f"    图像尺寸统计 (基于{len(sample_sizes)}个样本):")
        print(f"      宽度: {min(widths)}-{max(widths)} (平均: {np.mean(widths):.0f})")
        print(f"      高度: {min(heights)}-{max(heights)} (平均: {np.mean(heights):.0f})")
    
    # 6. 数据质量检查
    print("\n6. 数据质量检查:")
    for name, df in datasets.items():
        print(f"  {name}:")
        
        # 检查缺失值
        missing_text = df['text'].isnull().sum()
        missing_path = df['path'].isnull().sum()
        missing_label = df['label'].isnull().sum()
        print(f"    缺失值: text({missing_text}), path({missing_path}), label({missing_label})")
        
        # 检查重复行
        duplicates = df.duplicated().sum()
        print(f"    重复行: {duplicates} 条")
        
        # 检查空文本
        empty_text = (df['text'].str.strip() == '').sum()
        print(f"    空文本: {empty_text} 条")
        
        # 检查标签值
        unique_labels = sorted(df['label'].unique())
        print(f"    标签值: {unique_labels}")
    
    print("\n=" * 60)
    print("数据集分析完成!")
    print("=" * 60)
    
    # 返回分析结果摘要
    return {
        'total_samples': total_samples,
        'datasets': {name: len(df) for name, df in datasets.items()},
        'label_distribution': dict(total_labels),
        'missing_images': len(missing_images),
        'corrupted_images': len(corrupted_images),
        'image_count': image_count
    }

if __name__ == '__main__':
    results = analyze_dataset()
    print(f"\n分析结果摘要: {results}")