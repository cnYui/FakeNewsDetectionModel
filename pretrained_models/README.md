# 预训练模型下载指南

本项目需要以下预训练模型，请按照说明下载并放置到对应目录。

## 必需模型

### 1. BERT中文模型
- **模型名称**: `google-bert/bert-base-chinese`
- **大小**: ~400MB
- **用途**: 中文文本特征提取
- **放置目录**: `bert-base-chinese/`
- **下载地址**: https://huggingface.co/google-bert/bert-base-chinese

**目录结构**:
```
bert-base-chinese/
├── config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt
```

### 2. CLIP模型
- **模型名称**: `openai/clip-vit-base-patch32`
- **大小**: ~600MB
- **用途**: 图像-文本多模态特征提取
- **放置目录**: `clip-vit-base-patch32/`
- **下载地址**: https://huggingface.co/openai/clip-vit-base-patch32

**目录结构**:
```
clip-vit-base-patch32/
├── config.json
├── preprocessor_config.json
├── pytorch_model.bin
├── tokenizer_config.json
├── tokenizer.json
└── vocab.json
```

### 3. ResNet-18模型
- **模型名称**: `microsoft/resnet-18`
- **大小**: ~45MB
- **用途**: 图像特征提取
- **放置目录**: `resnet-18/`
- **下载地址**: https://huggingface.co/microsoft/resnet-18

**目录结构**:
```
resnet-18/
├── config.json
├── preprocessor_config.json
└── pytorch_model.bin
```

## 下载方法

### 方法1: 使用git lfs (推荐)
```bash
# 安装git lfs
git lfs install

# 下载BERT模型
cd pretrained_models/
git clone https://huggingface.co/google-bert/bert-base-chinese

# 下载CLIP模型
git clone https://huggingface.co/openai/clip-vit-base-patch32

# 下载ResNet模型
git clone https://huggingface.co/microsoft/resnet-18
```

### 方法2: 使用huggingface_hub
```python
from huggingface_hub import snapshot_download

# 下载BERT模型
snapshot_download(repo_id="google-bert/bert-base-chinese", 
                 local_dir="./bert-base-chinese")

# 下载CLIP模型
snapshot_download(repo_id="openai/clip-vit-base-patch32", 
                 local_dir="./clip-vit-base-patch32")

# 下载ResNet模型
snapshot_download(repo_id="microsoft/resnet-18", 
                 local_dir="./resnet-18")
```

### 方法3: 手动下载
访问上述Hugging Face链接，手动下载所有文件并按照目录结构放置。

## 验证下载

下载完成后，请确保目录结构如下：
```
pretrained_models/
├── README.md
├── bert-base-chinese/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
├── clip-vit-base-patch32/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── resnet-18/
    ├── config.json
    ├── pytorch_model.bin
    └── ...
```

## 注意事项

1. **网络要求**: 下载需要稳定的网络连接
2. **存储空间**: 确保有足够的磁盘空间（总计约1.1GB）
3. **权限**: 确保有写入权限
4. **版本兼容**: 使用最新版本的transformers库

如有问题，请参考项目文档或提交issue。