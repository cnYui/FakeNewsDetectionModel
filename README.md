# 多模态虚假新闻检测系统

一个基于深度学习的多模态虚假新闻检测系统，结合文本和图像特征进行虚假新闻识别。

## 项目概述

本项目实现了一个先进的多模态虚假新闻检测系统，通过融合文本和图像特征来识别虚假新闻。系统采用了BERT中文模型处理文本，CLIP模型处理图像，并通过Transformer架构进行多模态特征融合。

### 核心特性
- **多模态融合**: 结合文本和图像特征进行综合判断
- **中文优化**: 专门针对中文新闻文本进行优化
- **深度学习**: 基于BERT、CLIP等先进预训练模型
- **端到端训练**: 支持完整的训练和推理流程
- **高性能**: GPU加速训练，支持大规模数据集

## 项目结构

```
├── README.md                    # 项目主文档
├── TODO_REPRODUCTION.md         # 复现进度跟踪
├── .gitattributes              # Git属性配置
├── .gitignore                  # Git忽略文件
├── configs/                    # 配置文件目录
├── docs/                       # 文档目录
│   ├── README.md              # 详细项目文档
│   └── 多模态虚假新闻检测模型技术文档.md
├── pretrained_models/          # 预训练模型存放目录
│   ├── bert-base-chinese/     # BERT中文模型 (已下载)
│   ├── clip-vit-base-patch32/ # CLIP模型 (已下载)
│   └── resnet-18/             # ResNet-18模型
├── scripts/                    # 脚本目录
│   ├── download_bert_chinese.py    # BERT模型下载脚本
│   ├── download_clip_model.py      # CLIP模型下载脚本
│   ├── install_dependencies.sh     # 依赖安装脚本
│   ├── load_pretrained_text_model.py
│   └── quick_start_reproduction.sh # 快速复现脚本
└── src/                        # 源代码目录
    ├── __init__.py
    ├── data_processing/        # 数据处理模块
    │   ├── __init__.py
    │   ├── chinese_text_augmentation.py
    │   └── text_augmentation.py
    ├── models/                 # 模型定义
    │   ├── __init__.py
    │   ├── MultiModalFakeNewsDetector.py  # 主检测器
    │   ├── TransformerFusionModel.py      # 融合模型
    │   ├── clip_loader.py              # CLIP模型加载器
    │   ├── download_nltk_data.py       # NLTK数据下载
    │   ├── inference.py                # 推理接口
    │   ├── train_fusion_model.py       # 训练脚本
    │   ├── image_models/       # 图像模型
    │   │   ├── ImageProcessingModel.py
    │   │   └── SimpleImageModel.py
    │   └── text_models/        # 文本模型
    │       ├── TextProcessingModel.py
    │       ├── WordClassification.py
    │       ├── evaluate_model.py
    │       └── train_bert_only.py
    ├── training/               # 训练相关
    │   └── __init__.py
    └── utils/                  # 工具函数
        └── __init__.py
```

## 环境要求

### 系统要求
- **操作系统**: Linux (推荐 Ubuntu 18.04+)
- **Python**: 3.8+ (当前环境: Python 3.10.8)
- **GPU**: NVIDIA GPU with CUDA 11.0+ (当前: CUDA 12.1)
- **内存**: 16GB+ RAM 推荐
- **存储**: 10GB+ 可用空间

### 已安装依赖
当前环境已安装以下核心依赖：
- `torch==2.1.2+cu121` - PyTorch深度学习框架
- `transformers==4.51.3` - Hugging Face Transformers
- `numpy==1.26.3` - 数值计算
- `pandas==2.2.3` - 数据处理
- `pillow==10.2.0` - 图像处理
- `scikit-learn==1.6.1` - 机器学习工具
- `matplotlib==3.8.2` - 数据可视化
- `seaborn==0.13.2` - 统计可视化
- `jieba==0.42.1` - 中文分词
- `tqdm==4.64.1` - 进度条

## 快速开始

### 1. 环境验证
```bash
# 检查Python版本
python --version

# 检查CUDA可用性
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. 预训练模型状态
预训练模型已成功下载并验证：
- ✅ **BERT中文模型**: `/root/autodl-tmp/model_cache_new/bert-base-chinese/`
- ✅ **CLIP模型**: `/root/autodl-tmp/model_cache_new/clip-vit-base-patch32/`

模型加载测试通过：
```bash
# 验证模型加载
python -c "
from transformers import BertModel, CLIPModel
bert = BertModel.from_pretrained('/root/autodl-tmp/model_cache_new/bert-base-chinese')
clip = CLIPModel.from_pretrained('/root/autodl-tmp/model_cache_new/clip-vit-base-patch32')
print(f'BERT hidden size: {bert.config.hidden_size}')
print(f'CLIP vision hidden size: {clip.config.vision_config.hidden_size}')
"
```

### 3. 数据集准备
数据集位于 `/root/autodl-tmp/data/`，包含：
- `train.csv` - 训练集 (9,740条记录)
- `val.csv` - 验证集 (1,083条记录)  
- `test.csv` - 测试集 (2,454条记录)
- `images/` - 图像文件目录

数据格式：
```csv
path,text,label
./data/images/image_name.jpg,新闻文本内容,0/1
```

### 4. 模型训练
```bash
# 进入模型目录
cd /root/models/src/models

# 开始训练
python train_fusion_model.py \
    --data_dir /root/autodl-tmp/data \
    --model_cache_dir /root/autodl-tmp/model_cache_new \
    --images_dir /root/autodl-tmp/data/images \
    --batch_size 16 \
    --epochs 10 \
    --learning_rate 2e-5
```

### 5. 模型推理
```bash
# 单样本推理
python inference.py \
    --text "新闻文本内容" \
    --image_path "/path/to/image.jpg" \
    --model_path "/path/to/trained/model.pth"

# 批量推理
python MultiModalFakeNewsDetector.py
```

## 核心模块

### 1. 文本处理模块
- **BERT中文编码**: 使用 `bert-base-chinese` 进行文本特征提取
- **中文分词**: 基于 jieba 的中文文本预处理
- **文本增强**: 支持同义词替换、随机删除等数据增强技术

### 2. 图像处理模块
- **CLIP视觉编码**: 使用 `clip-vit-base-patch32` 进行图像特征提取
- **图像预处理**: 标准化、缩放、数据增强
- **多尺度特征**: 支持不同分辨率的图像输入

### 3. 多模态融合模块
- **Transformer融合**: 基于注意力机制的特征融合
- **跨模态对齐**: 文本和图像特征的语义对齐
- **特征交互**: 深度特征交互和信息整合

### 4. 训练与评估
- **端到端训练**: 支持多模态联合训练
- **评估指标**: 准确率、精确率、召回率、F1分数
- **模型保存**: 支持模型检查点和最佳模型保存

## 技术架构

```
输入层
├── 文本输入 → BERT编码器 → 文本特征 (768维)
└── 图像输入 → CLIP编码器 → 图像特征 (512维)
                    ↓
              特征融合层
         (Transformer + 注意力机制)
                    ↓
               分类器层
            (全连接 + Dropout)
                    ↓
              输出 (真/假)
```

## 性能指标

### 数据集统计
- **总样本数**: 13,277条
- **训练集**: 9,740条 (73.4%)
- **验证集**: 1,083条 (8.2%)
- **测试集**: 2,454条 (18.5%)
- **标签分布**: 假新闻约占60%，真新闻约占40%

### 模型性能 (预期)
- **准确率**: >85%
- **F1分数**: >0.83
- **训练时间**: ~2-4小时 (单GPU)
- **推理速度**: ~100ms/样本

## 使用示例

### Python API
```python
from src.models.MultiModalFakeNewsDetector import MultiModalFakeNewsDetector

# 初始化检测器
detector = MultiModalFakeNewsDetector(
    model_cache_dir='/root/autodl-tmp/model_cache_new'
)

# 加载训练好的模型
detector.load_model('path/to/trained/model.pth')

# 进行预测
result = detector.predict(
    text="新闻文本内容",
    image_path="/path/to/image.jpg"
)

print(f"预测结果: {'假新闻' if result['prediction'] == 1 else '真新闻'}")
print(f"置信度: {result['confidence']:.3f}")
```

### 命令行工具
```bash
# 训练模型
python src/models/train_fusion_model.py \
    --data_dir /root/autodl-tmp/data \
    --epochs 10 \
    --batch_size 16

# 评估模型
python src/models/evaluate_model.py \
    --model_path /path/to/model.pth \
    --test_data /root/autodl-tmp/data/test.csv

# 单样本推理
python src/models/inference.py \
    --text "新闻文本" \
    --image "/path/to/image.jpg"
```

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```bash
   # 减小批次大小
   python train_fusion_model.py --batch_size 8
   ```

2. **模型加载失败**
   ```bash
   # 检查模型路径
   ls -la /root/autodl-tmp/model_cache_new/
   ```

3. **图像文件缺失**
   ```bash
   # 检查图像目录
   ls -la /root/autodl-tmp/data/images/ | head -10
   ```

### 性能优化

- **混合精度训练**: 使用 `--fp16` 标志
- **梯度累积**: 使用 `--gradient_accumulation_steps`
- **数据并行**: 多GPU训练支持

## 项目状态

当前复现进度：
- ✅ 环境配置完成
- ✅ 预训练模型下载完成
- ✅ 数据集准备完成
- ✅ 代码路径配置完成
- 🔄 模型训练进行中
- ⏳ 模型评估待完成
- ⏳ 性能优化待完成

详细进度请查看 `TODO_REPRODUCTION.md`

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。

## 许可证

MIT License

## 联系方式

如有问题，请通过GitHub Issues联系。