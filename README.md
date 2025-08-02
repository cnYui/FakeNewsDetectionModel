# 多模态虚假新闻检测系统

一个基于深度学习的多模态虚假新闻检测系统，结合文本和图像特征进行虚假新闻识别。

## 项目结构

```
├── README.md                    # 项目主文档
├── .gitattributes              # Git属性配置
├── .DS_Store                   # macOS系统文件
├── __pycache__/                # Python缓存文件
├── configs/                    # 配置文件目录
├── docs/                       # 文档目录
│   ├── README.md              # 详细项目文档
│   └── 多模态虚假新闻检测模型技术文档.md
├── feature_cache/              # 特征缓存目录
├── model_cache/                # 模型缓存目录
├── pretrained_models/          # 预训练模型存放目录
│   ├── bert-base-chinese/     # BERT中文模型 (需要下载)
│   ├── clip-vit-base-patch32/ # CLIP模型 (需要下载)
│   └── resnet-18/             # ResNet-18模型 (需要下载)
├── scripts/                    # 脚本目录
│   ├── download_bert_chinese.py
│   ├── download_clip_model.py
│   ├── load_pretrained_text_model.py
│   └── install_dependencies.sh
└── src/                        # 源代码目录
    ├── data_processing/        # 数据处理模块
    │   ├── chinese_text_augmentation.py
    │   └── text_augmentation.py
    ├── models/                 # 模型定义
    │   ├── MultiModalFakeNewsDetector.py
    │   ├── TransformerFusionModel.py
    │   ├── clip_loader.py
    │   ├── download_nltk_data.py
    │   ├── inference.py
    │   ├── train_fusion_model.py
    │   ├── image_models/       # 图像模型
    │   │   ├── ImageProcessingModel.py
    │   │   └── SimpleImageModel.py
    │   └── text_models/        # 文本模型
    │       ├── TextProcessingModel.py
    │       ├── WordClassification.py
    │       ├── evaluate_model.py
    │       └── train_bert_only.py
    ├── training/               # 训练相关
    └── utils/                  # 工具函数
```

## 快速开始

### 1. 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.0+ (可选，用于GPU加速)

### 2. 安装依赖
```bash
bash scripts/install_dependencies.sh
```

### 3. 下载预训练模型
请将以下预训练模型下载并放置到 `pretrained_models/` 目录：

- **BERT中文模型**: `google-bert/bert-base-chinese` (~400MB)
- **CLIP模型**: `openai/clip-vit-base-patch32` (~600MB)  
- **ResNet-18模型**: `microsoft/resnet-18` (~45MB)

### 4. 运行主程序
```bash
cd src/models
python MultiModalFakeNewsDetector.py
```

## 主要功能

- **多模态融合**: 结合文本和图像特征
- **深度学习模型**: 基于BERT、CLIP和ResNet的先进架构
- **数据增强**: 支持文本和图像数据增强
- **模型训练**: 完整的训练和评估流程
- **推理接口**: 便于集成的推理API

## 技术特点

- 使用Transformer架构进行多模态特征融合
- 支持中文文本处理和增强
- 内存优化和GPU加速支持
- 模块化设计，易于扩展

## 文档

详细的技术文档请参考 `docs/` 目录中的文件。

## 许可证

MIT License