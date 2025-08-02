# 多模态虚假新闻检测模型

## 项目概述

本项目实现了一个基于多模态特征融合的虚假新闻检测模型，通过结合文本和图像两种不同模态的信息，利用深度学习技术提升虚假新闻检测的准确性和鲁棒性。

模型采用多模态融合架构，主要包含以下核心模块：
- **文本特征提取模块**：基于BERT和CLIP文本编码器
- **图像特征提取模块**：基于ResNet和CLIP视觉编码器  
- **特征融合模块**：基于Transformer的跨模态融合
- **分类模块**：基于MLP的最终分类器

## 🚀 快速开始

### 环境要求

- Python 3.9+
- PyTorch 1.10+
- CUDA支持（推荐）或Apple Silicon GPU (MPS)
- 至少2GB可用存储空间

### 安装依赖

```bash
# 安装基础依赖
pip install torch torchvision transformers pandas scikit-learn matplotlib seaborn tqdm pillow

# 或使用安装脚本
./install_dependencies.sh
```

## 📋 所需预训练模型

**重要提示**：以下三个预训练模型是项目运行的必需组件，由于在云服务器环境中，这些模型将由用户在本地下载后上传：

### 1. BERT中文模型 (`google-bert/bert-base-chinese`)
- **用途**：中文文本处理和特征提取
- **大小**：~400MB
- **保存路径**：`/root/models/model_cache/bert-base-chinese/`

### 2. CLIP模型 (`openai/clip-vit-base-patch32`)
- **用途**：图像和文本的多模态处理
- **大小**：~600MB  
- **保存路径**：`/root/models/model_cache/clip-vit-base-patch32/`

### 3. ResNet-18模型 (`microsoft/resnet-18`)
- **用途**：图像特征提取
- **大小**：~45MB
- **保存路径**：`/root/models/model_cache/resnet-18/`

### 模型文件结构

下载完成后，模型文件应保存在以下结构中：

```
model_cache/
├── bert-base-chinese/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── clip-vit-base-patch32/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── preprocessor_config.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   ├── merges.txt
│   └── vocab.json
└── resnet-18/
    ├── config.json
    └── pytorch_model.bin
```

## 🎯 使用方法

### 基本运行

```bash
# 运行主程序（小规模测试）
python MultiModalFakeNewsDetector.py
```

### 数据集准备

项目使用小规模MCFEND数据集子集进行快速验证：
- 训练集：`small_train.csv`
- 测试集：`small_test.csv`
- 验证集：`small_val.csv`
- 图像目录：包含对应的新闻配图

### 模型训练参数

当前实验设置（概念验证）：
- 最大样本数：10条数据
- 训练轮数：2个epoch
- 批次大小：2
- 数据划分：训练集8条，验证集2条

## 📊 模型性能

### 小规模测试结果
- **准确率**：100%
- **F1 Score**：100%
- **验证损失**：0.1699

### 完整数据集测试结果（历史）
- **准确率**：88.5%
- **精确率**：88.92%
- **召回率**：88.5%
- **F1分数**：88.48%
- **ROC AUC**：94.42%
- **平均精确率**：94.61%

## 🏗️ 项目架构

```
多模态假新闻检测系统
├── 数据处理模块
│   ├── 数据加载与预处理
│   ├── 数据增强
│   └── 数据集分割
├── 特征提取模块
│   ├── 文本特征提取 (BERT + CLIP文本编码)
│   └── 图像特征提取 (ResNet + CLIP视觉编码)
├── 模型融合模块
│   ├── Transformer融合
│   └── 特征拼接融合
└── 训练与评估模块
    ├── 模型训练
    ├── 性能评估
    └── 结果可视化
```

## 📁 项目文件结构

```
/root/models/
├── MultiModalFakeNewsDetector.py          # 主程序入口
├── chinese_text_augmentation.py           # 中文文本增强
├── text_augmentation.py                   # 文本增强工具
├── download_bert_chinese.py               # BERT模型下载脚本
├── download_clip_model.py                 # CLIP模型下载脚本
├── load_pretrained_text_model.py          # 预训练模型加载
├── install_dependencies.sh                # 依赖安装脚本
├── model_cache/                           # 模型缓存目录
├── feature_cache/                         # 特征缓存目录
├── fusionModelsnew/                       # 融合模型相关
│   ├── TransformerFusionModel.py
│   ├── train_fusion_model.py
│   ├── inference.py
│   └── clip_loader.py
├── testWordModel/                         # 文本模型测试
│   ├── TextProcessingModel.py
│   ├── WordClassification.py
│   ├── train_bert_only.py
│   └── evaluate_model.py
├── testPictureModel/                      # 图像模型测试
│   ├── ImageProcessingModel.py
│   └── SimpleImageModel.py
├── past/                                  # 历史文档
│   ├── README.md
│   ├── README-testonly.md
│   ├── README_NEW.md
│   ├── MODEL_DOWNLOAD_GUIDE.md
│   └── project_architecture.md
└── 多模态虚假新闻检测模型技术文档.md        # 详细技术文档
```

## 🔧 高级功能

### 文本增强
- 同义词替换
- 随机插入、交换、删除
- 回译技术
- 中文特定增强策略

### 图像增强
- 随机裁剪、翻转、旋转
- 颜色抖动、灰度化
- 模糊处理
- Mixup和CutMix技术

### 对抗训练
- FGM (Fast Gradient Method) 对抗训练
- 增强模型鲁棒性

## 📈 评估指标

模型使用以下指标进行评估：
- 准确率 (Accuracy)
- 精确率 (Precision)
- 召回率 (Recall)
- F1值 (F1 Score)
- ROC曲线和AUC值
- 精确率-召回率曲线
- 混淆矩阵

## 🛠️ 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件是否完整
   - 确认模型路径设置正确
   - 验证网络连接（如需在线下载）

2. **内存不足**
   - 减少批次大小
   - 启用梯度累积
   - 使用混合精度训练

3. **CUDA错误**
   - 检查CUDA版本兼容性
   - 确认GPU内存充足
   - 可切换到CPU模式

### 性能优化

- 使用特征缓存加速训练
- 启用混合精度训练 (AMP)
- 合理设置数据加载器工作线程数
- 定期进行垃圾回收

## 📚 技术文档

详细的技术实现和实验结果请参考：
- [多模态虚假新闻检测模型技术文档.md](./多模态虚假新闻检测模型技术文档.md)
- [历史文档](./past/)

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

感谢以下开源项目的支持：
- [Transformers](https://github.com/huggingface/transformers) - Hugging Face
- [PyTorch](https://pytorch.org/) - Facebook AI Research
- [CLIP](https://github.com/openai/CLIP) - OpenAI
- [BERT](https://github.com/google-research/bert) - Google Research

---

**注意**：本项目为研究和教育目的，实际应用时请根据具体需求进行调整和优化。