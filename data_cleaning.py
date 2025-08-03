#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗脚本
用于清洗假新闻检测数据集，处理缺失数据、重复数据、异常数据等问题
"""

import pandas as pd
import numpy as np
import os
import argparse
from PIL import Image
import shutil
from collections import Counter
import re

class FakeNewsDataCleaner:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.train_path = os.path.join(data_dir, 'train.csv')
        self.val_path = os.path.join(data_dir, 'val.csv')
        self.test_path = os.path.join(data_dir, 'test.csv')
        
        # 清洗后的数据保存路径
        self.cleaned_dir = os.path.join(data_dir, 'cleaned')
        os.makedirs(self.cleaned_dir, exist_ok=True)
        
    def load_data(self):
        """加载原始数据"""
        print("加载原始数据...")
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)
        
        print(f"训练集: {len(self.train_df)} 条")
        print(f"验证集: {len(self.val_df)} 条")
        print(f"测试集: {len(self.test_df)} 条")
        
        # 合并数据进行全局分析
        self.all_df = pd.concat([
            self.train_df.assign(split='train'),
            self.val_df.assign(split='val'),
            self.test_df.assign(split='test')
        ], ignore_index=True)
        
        print(f"总计: {len(self.all_df)} 条")
        
    def analyze_data_quality(self):
        """分析数据质量"""
        print("\n=== 数据质量分析 ===")
        
        # 缺失值分析
        print("\n缺失值统计:")
        missing_stats = self.all_df.isnull().sum()
        print(missing_stats)
        
        # 重复数据分析
        print("\n重复数据统计:")
        dup_by_path = self.all_df.duplicated(subset=['path']).sum()
        dup_by_text = self.all_df.duplicated(subset=['text']).sum()
        dup_by_all = self.all_df.duplicated().sum()
        print(f"按图片路径重复: {dup_by_path} 条")
        print(f"按文本重复: {dup_by_text} 条")
        print(f"完全重复: {dup_by_all} 条")
        
        # 文本质量分析
        print("\n文本质量分析:")
        self.all_df['text_length'] = self.all_df['text'].fillna('').str.len()
        empty_text = (self.all_df['text'].isnull() | (self.all_df['text'] == '')).sum()
        short_text = (self.all_df['text_length'] < 10).sum()
        long_text = (self.all_df['text_length'] > 1000).sum()
        
        print(f"空文本: {empty_text} 条")
        print(f"过短文本(<10字符): {short_text} 条")
        print(f"过长文本(>1000字符): {long_text} 条")
        
        # 图片质量分析
        print("\n图片质量分析:")
        self.check_image_quality()
        
        # 标签分布分析
        print("\n标签分布:")
        label_counts = self.all_df['label'].value_counts()
        print(label_counts)
        balance_ratio = label_counts.min() / label_counts.max()
        print(f"标签平衡度: {balance_ratio:.3f}")
        
    def check_image_quality(self):
        """检查图片质量"""
        print("检查图片质量...")
        corrupted_images = []
        missing_images = []
        valid_images = []
        
        for idx, row in self.all_df.iterrows():
            img_path = row['path']
            if pd.isna(img_path):
                missing_images.append(idx)
                continue
                
            # 处理路径
            if img_path.startswith('./data/'):
                img_path = img_path[7:]
            
            full_path = os.path.join(self.data_dir, img_path)
            
            if not os.path.exists(full_path):
                missing_images.append(idx)
                continue
                
            try:
                with Image.open(full_path) as img:
                    img.load()  # 尝试加载图片数据
                    valid_images.append(idx)
            except Exception as e:
                corrupted_images.append((idx, str(e)))
        
        print(f"有效图片: {len(valid_images)} 张")
        print(f"缺失图片: {len(missing_images)} 张")
        print(f"损坏图片: {len(corrupted_images)} 张")
        
        self.valid_image_indices = set(valid_images)
        self.missing_image_indices = set(missing_images)
        self.corrupted_image_indices = set([idx for idx, _ in corrupted_images])
        
    def clean_data(self, remove_duplicates=True, remove_missing_text=True, 
                   remove_short_text=True, remove_missing_images=True,
                   remove_corrupted_images=True, min_text_length=10):
        """清洗数据"""
        print("\n=== 开始数据清洗 ===")
        
        cleaned_df = self.all_df.copy()
        initial_count = len(cleaned_df)
        
        # 1. 移除缺失文本
        if remove_missing_text:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df.dropna(subset=['text'])
            cleaned_df = cleaned_df[cleaned_df['text'] != '']
            after_count = len(cleaned_df)
            print(f"移除缺失文本: {before_count - after_count} 条")
        
        # 2. 移除过短文本
        if remove_short_text:
            before_count = len(cleaned_df)
            cleaned_df['text_length'] = cleaned_df['text'].str.len()
            cleaned_df = cleaned_df[cleaned_df['text_length'] >= min_text_length]
            after_count = len(cleaned_df)
            print(f"移除过短文本(<{min_text_length}字符): {before_count - after_count} 条")
        
        # 3. 移除缺失图片
        if remove_missing_images:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df[~cleaned_df.index.isin(self.missing_image_indices)]
            after_count = len(cleaned_df)
            print(f"移除缺失图片: {before_count - after_count} 条")
        
        # 4. 移除损坏图片
        if remove_corrupted_images:
            before_count = len(cleaned_df)
            cleaned_df = cleaned_df[~cleaned_df.index.isin(self.corrupted_image_indices)]
            after_count = len(cleaned_df)
            print(f"移除损坏图片: {before_count - after_count} 条")
        
        # 5. 移除重复数据
        if remove_duplicates:
            before_count = len(cleaned_df)
            # 优先保留训练集中的数据
            cleaned_df = cleaned_df.sort_values('split', key=lambda x: x.map({'train': 0, 'val': 1, 'test': 2}))
            cleaned_df = cleaned_df.drop_duplicates(subset=['text'], keep='first')
            after_count = len(cleaned_df)
            print(f"移除重复文本: {before_count - after_count} 条")
        
        # 6. 文本清洗
        print("进行文本清洗...")
        cleaned_df['text'] = cleaned_df['text'].apply(self.clean_text)
        
        final_count = len(cleaned_df)
        print(f"\n清洗完成: {initial_count} -> {final_count} 条 (移除 {initial_count - final_count} 条, {(initial_count - final_count)/initial_count*100:.2f}%)")
        
        self.cleaned_df = cleaned_df
        
    def clean_text(self, text):
        """清洗文本内容"""
        if pd.isna(text):
            return ''
        
        # 移除多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # 移除HTML实体
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&quot;', '"')
        
        return text
    
    def split_cleaned_data(self):
        """将清洗后的数据重新分割为训练集、验证集、测试集"""
        print("\n=== 重新分割数据集 ===")
        
        # 按原始分割重新分组
        train_cleaned = self.cleaned_df[self.cleaned_df['split'] == 'train'].copy()
        val_cleaned = self.cleaned_df[self.cleaned_df['split'] == 'val'].copy()
        test_cleaned = self.cleaned_df[self.cleaned_df['split'] == 'test'].copy()
        
        # 移除split列
        train_cleaned = train_cleaned.drop(['split', 'text_length'], axis=1, errors='ignore')
        val_cleaned = val_cleaned.drop(['split', 'text_length'], axis=1, errors='ignore')
        test_cleaned = test_cleaned.drop(['split', 'text_length'], axis=1, errors='ignore')
        
        print(f"清洗后训练集: {len(train_cleaned)} 条")
        print(f"清洗后验证集: {len(val_cleaned)} 条")
        print(f"清洗后测试集: {len(test_cleaned)} 条")
        
        # 检查标签分布
        print("\n清洗后标签分布:")
        for name, df in [('训练集', train_cleaned), ('验证集', val_cleaned), ('测试集', test_cleaned)]:
            if len(df) > 0:
                label_counts = df['label'].value_counts()
                print(f"{name}: {dict(label_counts)}")
        
        return train_cleaned, val_cleaned, test_cleaned
    
    def save_cleaned_data(self, train_cleaned, val_cleaned, test_cleaned):
        """保存清洗后的数据"""
        print("\n=== 保存清洗后的数据 ===")
        
        # 保存清洗后的CSV文件
        train_cleaned.to_csv(os.path.join(self.cleaned_dir, 'train_cleaned.csv'), index=False)
        val_cleaned.to_csv(os.path.join(self.cleaned_dir, 'val_cleaned.csv'), index=False)
        test_cleaned.to_csv(os.path.join(self.cleaned_dir, 'test_cleaned.csv'), index=False)
        
        print(f"清洗后的数据已保存到: {self.cleaned_dir}")
        print(f"- train_cleaned.csv: {len(train_cleaned)} 条")
        print(f"- val_cleaned.csv: {len(val_cleaned)} 条")
        print(f"- test_cleaned.csv: {len(test_cleaned)} 条")
        
        # 生成清洗报告
        self.generate_cleaning_report(train_cleaned, val_cleaned, test_cleaned)
    
    def generate_cleaning_report(self, train_cleaned, val_cleaned, test_cleaned):
        """生成清洗报告"""
        report_path = os.path.join(self.cleaned_dir, 'cleaning_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("数据清洗报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("原始数据统计:\n")
            f.write(f"训练集: {len(self.train_df)} 条\n")
            f.write(f"验证集: {len(self.val_df)} 条\n")
            f.write(f"测试集: {len(self.test_df)} 条\n")
            f.write(f"总计: {len(self.all_df)} 条\n\n")
            
            f.write("清洗后数据统计:\n")
            f.write(f"训练集: {len(train_cleaned)} 条\n")
            f.write(f"验证集: {len(val_cleaned)} 条\n")
            f.write(f"测试集: {len(test_cleaned)} 条\n")
            f.write(f"总计: {len(train_cleaned) + len(val_cleaned) + len(test_cleaned)} 条\n\n")
            
            f.write("数据质量问题:\n")
            f.write(f"缺失图片: {len(self.missing_image_indices)} 条\n")
            f.write(f"损坏图片: {len(self.corrupted_image_indices)} 条\n")
            f.write(f"缺失文本: {self.all_df['text'].isnull().sum()} 条\n")
            f.write(f"重复文本: {self.all_df.duplicated(subset=['text']).sum()} 条\n")
            
        print(f"清洗报告已保存到: {report_path}")
    
    def run_full_cleaning(self):
        """运行完整的数据清洗流程"""
        self.load_data()
        self.analyze_data_quality()
        self.clean_data()
        train_cleaned, val_cleaned, test_cleaned = self.split_cleaned_data()
        self.save_cleaned_data(train_cleaned, val_cleaned, test_cleaned)
        
        print("\n=== 数据清洗完成 ===")
        print(f"清洗后的数据保存在: {self.cleaned_dir}")
        print("建议使用清洗后的数据进行训练以获得更好的效果。")

def main():
    parser = argparse.ArgumentParser(description='假新闻数据集清洗工具')
    parser.add_argument('--data_dir', type=str, default='/root/autodl-tmp/data',
                       help='数据目录路径')
    parser.add_argument('--min_text_length', type=int, default=10,
                       help='最小文本长度')
    parser.add_argument('--keep_duplicates', action='store_true',
                       help='保留重复数据')
    parser.add_argument('--keep_missing_images', action='store_true',
                       help='保留缺失图片的记录')
    
    args = parser.parse_args()
    
    cleaner = FakeNewsDataCleaner(args.data_dir)
    
    # 设置清洗参数
    cleaner.load_data()
    cleaner.analyze_data_quality()
    cleaner.clean_data(
        remove_duplicates=not args.keep_duplicates,
        remove_missing_images=not args.keep_missing_images,
        min_text_length=args.min_text_length
    )
    
    train_cleaned, val_cleaned, test_cleaned = cleaner.split_cleaned_data()
    cleaner.save_cleaned_data(train_cleaned, val_cleaned, test_cleaned)

if __name__ == '__main__':
    main()