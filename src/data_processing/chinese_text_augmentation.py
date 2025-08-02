#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文文本数据增强模块
提供同义词替换等文本增强功能
"""

import random
import jieba
import numpy as np
from typing import List, Tuple, Dict, Optional, Union

# 简单的中文同义词词典
# 这里只是一个示例，实际应用中可以扩展这个词典或使用更大的同义词库
CHINESE_SYNONYM_DICT = {
    # 名词
    '新闻': ['报道', '消息', '资讯', '快讯'],
    '政府': ['当局', '官方', '行政机构', '政务部门'],
    '经济': ['财经', '金融', '商业', '产业'],
    '政策': ['方针', '措施', '规定', '法规'],
    '企业': ['公司', '商家', '厂商', '机构'],
    '市场': ['行情', '商场', '交易市场', '买卖'],
    '社会': ['民间', '群众', '公众', '民众'],
    '学校': ['院校', '校园', '学院', '教育机构'],
    '医院': ['医疗机构', '诊所', '卫生院', '保健中心'],
    '疫情': ['疫病', '传染病', '流行病', '瘟疫'],
    '病毒': ['病原体', '微生物', '细菌', '病菌'],
    '专家': ['学者', '权威', '研究员', '专业人士'],
    '研究': ['研发', '探索', '调查', '考察'],
    '数据': ['资料', '信息', '统计数字', '数值'],
    '问题': ['难题', '困难', '疑问', '状况'],
    '情况': ['状况', '形势', '局面', '现状'],
    '时间': ['时刻', '时候', '期间', '阶段'],
    '地区': ['区域', '地带', '区段', '片区'],
    '国家': ['国度', '国土', '国区', '邦国'],
    '人民': ['民众', '百姓', '公民', '群众'],
    '图片': ['照片', '图像', '相片', '影像'],
    '视频': ['影片', '录像', '短片', '影像'],
    '网络': ['互联网', '网站', '在线', '线上'],
    '科技': ['技术', '科学技术', '高科技', '尖端技术'],
    '产品': ['商品', '物品', '制品', '货品'],
    '服务': ['服务项目', '服务内容', '服务事项', '服务工作'],
    '价格': ['价钱', '售价', '费用', '金额'],
    '活动': ['行动', '举措', '行为', '动作'],
    '事件': ['事情', '事故', '案件', '状况'],
    '结果': ['后果', '效果', '成果', '结局'],
    
    # 动词
    '发布': ['公布', '宣布', '发表', '推出'],
    '表示': ['表达', '说明', '指出', '声明'],
    '认为': ['觉得', '以为', '看来', '认定'],
    '发现': ['发觉', '察觉', '找到', '看到'],
    '提高': ['增加', '提升', '增强', '加强'],
    '降低': ['减少', '下降', '减低', '下调'],
    '增长': ['增加', '提高', '上升', '攀升'],
    '下降': ['减少', '降低', '下跌', '回落'],
    '实施': ['执行', '推行', '开展', '进行'],
    '推动': ['促进', '推进', '推广', '促使'],
    '影响': ['作用', '效应', '波及', '冲击'],
    '分析': ['研究', '探讨', '剖析', '解析'],
    '调查': ['考察', '研究', '探索', '调研'],
    '报道': ['报导', '报告', '叙述', '记述'],
    '解决': ['处理', '解答', '应对', '消除'],
    '支持': ['支撑', '拥护', '赞同', '帮助'],
    '反对': ['抵制', '抗议', '否决', '不同意'],
    '使用': ['应用', '利用', '采用', '运用'],
    '开展': ['展开', '进行', '开始', '启动'],
    '参与': ['加入', '参加', '介入', '参预'],
    '关注': ['注意', '重视', '留意', '关心'],
    '发生': ['出现', '产生', '发作', '爆发'],
    '出现': ['呈现', '显现', '涌现', '露面'],
    '提供': ['供给', '给予', '供应', '呈现'],
    '获得': ['取得', '得到', '获取', '赢得'],
    '进行': ['开展', '实施', '举行', '展开'],
    '加强': ['增强', '提高', '强化', '增进'],
    '保持': ['维持', '保留', '保存', '维护'],
    '促进': ['推动', '推进', '推广', '促使'],
    '预防': ['防止', '避免', '防范', '预先防备'],
    
    # 形容词
    '重要': ['关键', '主要', '重大', '关键性'],
    '严重': ['严峻', '严厉', '严格', '严肃'],
    '积极': ['主动', '热情', '活跃', '踊跃'],
    '消极': ['被动', '负面', '悲观', '低沉'],
    '有效': ['有用', '有力', '有益', '有功效'],
    '无效': ['无用', '无力', '无益', '无功效'],
    '正确': ['正当', '正式', '正规', '正当'],
    '错误': ['错位', '错落', '错乱', '错杂'],
    '良好': ['优良', '优质', '优秀', '优异'],
    '恶劣': ['糟糕', '不良', '坏', '差'],
    '安全': ['平安', '安稳', '安定', '安宁'],
    '危险': ['危急', '险恶', '险峻', '凶险'],
    '健康': ['健全', '强健', '康健', '健壮'],
    '疾病': ['病症', '病态', '疾患', '病痛'],
    '快速': ['迅速', '快捷', '迅猛', '神速'],
    '缓慢': ['慢慢', '迟缓', '缓行', '迟滞'],
    '简单': ['简易', '简略', '简便', '简明'],
    '复杂': ['繁杂', '繁复', '纷繁', '错综复杂'],
    '真实': ['真正', '真切', '真诚', '真挚'],
    '虚假': ['虚构', '虚伪', '虚幻', '虚无'],
    '清晰': ['明确', '明晰', '清楚', '明白'],
    '模糊': ['含糊', '朦胧', '不清', '不明'],
    '丰富': ['充足', '充实', '充裕', '充分'],
    '贫乏': ['缺乏', '不足', '匮乏', '短缺'],
    '美丽': ['漂亮', '美好', '美观', '美妙'],
    '丑陋': ['难看', '丑恶', '丑怪', '丑态'],
    '成功': ['胜利', '成就', '成绩', '成效'],
    '失败': ['失利', '挫折', '失误', '败北'],
    '正常': ['正规', '常规', '常态', '正当'],
    '异常': ['不正常', '反常', '不寻常', '特殊'],
}

class ChineseTextAugmenter:
    """
    中文文本数据增强类
    提供多种文本增强方法，如同义词替换、回译等
    """
    def __init__(self, 
                 synonym_prob=0.3,  # 同义词替换概率
                 synonym_percent=0.2,  # 替换词汇比例
                 random_state=42):
        """
        初始化文本增强器
        
        Args:
            synonym_prob: 进行同义词替换的概率
            synonym_percent: 替换文本中词汇的比例
            random_state: 随机种子
        """
        self.synonym_prob = synonym_prob
        self.synonym_percent = synonym_percent
        self.random_state = random_state
        
        # 设置随机种子
        random.seed(random_state)
        np.random.seed(random_state)
        
        # 停用词列表 (常见的不需要替换的词)
        self.stopwords = set(['的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '你', '我', '他', '她', '它', '们'])
        
        # 同义词词典
        self.synonym_dict = CHINESE_SYNONYM_DICT
        
        print("中文文本增强器初始化完成")
    
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
        
        # 过滤停用词和不在同义词词典中的词
        candidate_words = []
        candidate_indices = []
        
        for i, word in enumerate(words):
            if word not in self.stopwords and word in self.synonym_dict:
                candidate_words.append(word)
                candidate_indices.append(i)
        
        # 如果没有合适的词，返回原文本
        if not candidate_words:
            return text
        
        # 确定要替换的词数量
        n_replace = max(1, int(len(candidate_words) * self.synonym_percent))
        n_replace = min(n_replace, len(candidate_words))  # 确保不超过候选词数量
        
        # 随机选择要替换的词的索引
        replace_indices = random.sample(range(len(candidate_words)), n_replace)
        
        # 替换词
        for idx in replace_indices:
            word = candidate_words[idx]
            position = candidate_indices[idx]
            
            # 获取同义词列表
            synonyms_list = self.synonym_dict.get(word, [])
            
            # 如果有同义词，进行替换
            if synonyms_list:
                synonym = random.choice(synonyms_list)
                words[position] = synonym
        
        # 重新组合文本
        augmented_text = ''.join(words)
        
        return augmented_text
    
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
    augmenter = ChineseTextAugmenter(synonym_prob=1.0, synonym_percent=0.5)
    
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
