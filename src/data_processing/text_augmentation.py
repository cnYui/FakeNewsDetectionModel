#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本数据增强模块
提供同义词替换等文本增强功能
"""

import random
import jieba
import synonyms
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

class TextAugmenter:
    """
    文本数据增强类
    提供多种文本增强方法，如同义词替换、回译等
    """
    def __init__(self, 
                 synonym_prob=0.3,  # 同义词替换概率
                 synonym_percent=0.2,  # 替换词汇比例
                 min_similarity=0.7,  # 最小相似度
                 random_state=42):
        """
        初始化文本增强器
        
        Args:
            synonym_prob: 进行同义词替换的概率
            synonym_percent: 替换文本中词汇的比例
            min_similarity: 同义词最小相似度阈值
            random_state: 随机种子
        """
        self.synonym_prob = synonym_prob
        self.synonym_percent = synonym_percent
        self.min_similarity = min_similarity
        self.random_state = random_state
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 停用词列表 (常见的不需要替换的词)
        self.stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '你', '我', '他', '她', '它', '们'])
        
        print("文本增强器初始化完成")
    
    def synonym_replacement(self, text: str) -> str:
        """
        同义词替换增强
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        # 随机决定是否进行同义词替换
        if random.random() >= self.synonym_prob:
            return text
        
        # 分词
        words = list(jieba.cut(text))
        
        # 过滤停用词
        candidate_words = [word for word in words if word not in self.stopwords and len(word) > 1]
        
        # 如果没有合适的词，返回原文本
        if not candidate_words:
            return text
        
        # 确定要替换的词数量
        n_replace = max(1, int(len(candidate_words) * self.synonym_percent))
        n_replace = min(n_replace, len(candidate_words))  # 确保不超过候选词数量
        
        # 随机选择要替换的词的索引
        replace_indices = random.sample(range(len(candidate_words)), n_replace)
        
        # 获取要替换的词
        words_to_replace = [candidate_words[i] for i in replace_indices]
        
        # 替换词
        for word in words_to_replace:
            # 找出词在原始分词列表中的位置
            word_indices = [i for i, w in enumerate(words) if w == word]
            
            if not word_indices:
                continue
                
            # 随机选择一个位置进行替换
            replace_idx = random.choice(word_indices)
            
            # 获取同义词
            synonyms_list = self._get_synonyms(word)
            
            # 如果有同义词，进行替换
            if synonyms_list:
                synonym = random.choice(synonyms_list)
                words[replace_idx] = synonym
        
        # 重新组合文本
        augmented_text = ''.join(words)
        
        return augmented_text
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        获取词的同义词列表
        
        Args:
            word: 输入词
            
        Returns:
            同义词列表
        """
        # 使用synonyms获取同义词
        try:
            word_synonyms = []
            # 获取同义词及其相似度
            synonyms_list = synonyms.nearby(word)
            
            if not synonyms_list or len(synonyms_list) < 2:
                return []
                
            words_list, scores_list = synonyms_list
            
            # 过滤低相似度的同义词
            for i, (syn, score) in enumerate(zip(words_list[1:], scores_list[1:])):
                if score >= self.min_similarity and syn != word:
                    word_synonyms.append(syn)
            
            return word_synonyms
        except Exception as e:
            print(f"获取同义词时出错: {e}")
            return []
    
    def augment(self, text: str) -> str:
        """
        对文本进行增强
        
        Args:
            text: 输入文本
            
        Returns:
            增强后的文本
        """
        return self.synonym_replacement(text)
    
    def batch_augment(self, texts: List[str], n_aug: int = 1) -> List[str]:
        """
        批量增强文本
        
        Args:
            texts: 输入文本列表
            n_aug: 每个文本增强的数量
            
        Returns:
            增强后的文本列表
        """
        augmented_texts = []
        
        for text in texts:
            # 添加原始文本
            augmented_texts.append(text)
            
            # 生成增强文本
            for _ in range(n_aug):
                augmented_text = self.augment(text)
                if augmented_text != text:  # 只添加与原文本不同的增强文本
                    augmented_texts.append(augmented_text)
        
        return augmented_texts

# 测试代码
if __name__ == "__main__":
    augmenter = TextAugmenter(synonym_prob=1.0, synonym_percent=0.5)
    
    test_texts = [
        "这条新闻是关于经济发展的重要报道",
        "政府宣布了新的政策措施来刺激经济增长",
        "科学家发现了一种新型病毒的传播途径"
    ]
    
    for text in test_texts:
        print(f"原文: {text}")
        augmented = augmenter.augment(text)
        print(f"增强: {augmented}")
        print("---")
    
    # 批量增强测试
    batch_results = augmenter.batch_augment(test_texts, n_aug=2)
    print(f"批量增强结果数量: {len(batch_results)}")
