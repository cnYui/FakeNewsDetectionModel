#!/bin/bash

# 激活conda环境
source /root/miniconda3/bin/activate fusion_model

# 安装PyTorch依赖项
pip install torch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0

# 安装其他依赖项
pip install -r /root/Github/models/testWordModel/requirements_other.txt

echo "所有依赖项安装完成！"
