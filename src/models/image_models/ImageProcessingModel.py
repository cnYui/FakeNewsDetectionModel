import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import models, transforms
from transformers import CLIPProcessor, CLIPVisionModel
import pandas as pd
from PIL import Image
import os
import json
import time
import gc
import signal
import socket
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import logging
import sys
from tqdm import tqdm
import platform
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import copy  # 添加导入copy模块

# 设置常量
CACHE_DIR = '/Users/wujianxiang/Documents/GitHub/models/model_cache'
DEFAULT_IMAGE_PATH = os.path.join(CACHE_DIR, 'default_image.jpg')
MAX_SAMPLES = 8000  # 默认最大样本数
SOCKET_TIMEOUT = 10  # 设置全局socket超时为10秒
MAX_RETRY = 3  # 最大重试次数
CLIP_MODEL_PATH = os.path.join(CACHE_DIR, 'clip-vit-base-patch32')  # 本地CLIP模型路径

# CUDA相关配置
CUDA_VISIBLE_DEVICES = "0"  # 使用的GPU编号，RTX 3090通常为0
CUDA_LAUNCH_BLOCKING = "1"  # 设置为1可以帮助调试CUDA错误
CUDA_BATCH_SIZE = 32  # CUDA环境下的默认批量大小
CUDA_NUM_WORKERS = 8  # CUDA环境下的数据加载器工作线程数

# 设置环境变量
os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
os.environ['CUDA_LAUNCH_BLOCKING'] = CUDA_LAUNCH_BLOCKING

# 设置socket超时
socket.setdefaulttimeout(SOCKET_TIMEOUT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 创建默认图像
def create_default_image(size=(224, 224)):
    """创建默认图像用于替代异常图像"""
    # 检查默认图像是否存在
    if os.path.exists(DEFAULT_IMAGE_PATH):
        try:
            return Image.open(DEFAULT_IMAGE_PATH).convert('RGB').resize(size)
        except:
            pass
    
    # 创建灰色图像
    img = Image.new('RGB', size, color=(128, 128, 128))
    
    # 保存默认图像
    os.makedirs(os.path.dirname(DEFAULT_IMAGE_PATH), exist_ok=True)
    img.save(DEFAULT_IMAGE_PATH)
    
    return img

class FakeNewsImageDataset(Dataset):
    def __init__(self, data_path, images_dir, clip_processor=None, transform=None, use_cache=True, max_samples=None, image_size=(224, 224), low_memory_mode=False, data_augmentation=False, verbose=False):
        """
        初始化图像数据集
        
        Args:
            data_path: 数据文件路径
            images_dir: 图像目录路径
            clip_processor: CLIP处理器
            transform: 图像变换（用于ResNet）
            use_cache: 是否使用缓存
            max_samples: 最大样本数，用于限制内存使用
            image_size: 图像尺寸，默认为(224, 224)
            low_memory_mode: 是否启用低内存模式
            data_augmentation: 是否使用数据增强
            verbose: 是否显示详细的处理信息
        """
        self.data_path = data_path
        self.images_dir = images_dir
        self.clip_processor = clip_processor
        self.use_cache = use_cache
        self.image_size = image_size
        self.low_memory_mode = low_memory_mode
        self.data_augmentation = data_augmentation
        self.verbose = verbose
        
        # 检查数据文件是否存在
        if not os.path.exists(data_path):
            print(f"错误：数据文件不存在：{data_path}")
            # 如果数据文件不存在，创建空数据集
            self.valid_indices = []
            self.image_paths = []
            self.df = None
            return
        
        # 检查图像目录是否存在
        if not os.path.exists(images_dir):
            print(f"错误：图像目录不存在：{images_dir}")
            # 如果图像目录不存在，创建空数据集
            self.valid_indices = []
            self.image_paths = []
            self.df = None
            return
        
        # 缓存文件路径
        cache_dir = os.path.join(os.path.dirname(data_path), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        cache_filename = os.path.basename(data_path).split('.')[0] + '_dataset_cache.pkl'
        self.cache_path = os.path.join(cache_dir, cache_filename)
        
        # 为ResNet设置默认变换
        if transform is None:
            if self.data_augmentation and 'train' in data_path.lower():
                # 训练集使用数据增强
                self.transform = transforms.Compose([
                    transforms.Resize((self.image_size[0] + 20, self.image_size[1] + 20)),  # 稍大一些以便进行裁剪
                    transforms.RandomCrop(self.image_size),  # 随机裁剪
                    transforms.RandomHorizontalFlip(),  # 随机水平翻转
                    transforms.RandomRotation(15),  # 随机旋转
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 颜色抖动
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                # 验证集和测试集不使用数据增强
                self.transform = transforms.Compose([
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform
            
        # 尝试从缓存加载，使用更高效的方式
        if self.use_cache and os.path.exists(self.cache_path):
            try:
                print(f"尝试从缓存加载数据集: {self.cache_path}")
                import pickle
                
                start_time = time.time()
                
                # 低内存模式下使用更节省内存的加载方式
                if self.low_memory_mode:
                    # 使用mmap_mode='r'以减少内存使用，仅在需要时从磁盘读取数据
                    import numpy as np
                    with open(self.cache_path, 'rb') as f:
                        # 只读取必要的元数据
                        cache_data = pickle.load(f)
                        self.valid_indices = cache_data['valid_indices']
                        self.image_paths = cache_data['image_paths']
                        
                        # 如果设置了最大样本数，则限制样本数量
                        if max_samples is not None and max_samples > 0 and len(self.valid_indices) > max_samples:
                            print(f"限制样本数量为 {max_samples} (原有 {len(self.valid_indices)} 条记录)")
                            self.valid_indices = self.valid_indices[:max_samples]
                            self.image_paths = self.image_paths[:max_samples]
                        
                        # 不加载完整的DataFrame，而是保存文件路径
                        self.df_path = data_path
                        self.df = None  # 不保存完整DataFrame
                else:
                    # 常规模式 - 完全加载到内存
                    with open(self.cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                        
                    # 分开赋值以提高性能
                    self.df = cache_data['df']
                    self.image_paths = cache_data['image_paths']
                    self.valid_indices = cache_data['valid_indices']
                    
                    # 如果设置了最大样本数，则限制样本数量
                    if max_samples is not None and max_samples > 0 and len(self.valid_indices) > max_samples:
                        print(f"限制样本数量为 {max_samples} (原有 {len(self.valid_indices)} 条记录)")
                        self.valid_indices = self.valid_indices[:max_samples]
                        self.image_paths = self.image_paths[:max_samples]
                    
                    # 触发垃圾回收以释放内存
                    del cache_data
                    gc.collect()
                
                load_time = time.time() - start_time
                print(f"成功从缓存加载数据: {len(self.valid_indices)} 条有效记录 (用时: {load_time:.2f}秒)")
                
                return
            except Exception as e:
                print(f"从缓存加载失败: {e}，将重新处理数据")
        
        # 如果缓存加载失败或不使用缓存，则处理原始数据
        try:
            # 使用新的数据加载方法
            self.df, self.valid_indices, self.image_paths = self._load_data_from_file(data_path, max_samples)
            
            # 如果没有有效的数据记录，则创建一个空的数据集
            if not self.valid_indices:
                print("警告：没有有效的数据记录，创建空数据集")
                self.valid_indices = []
                self.image_paths = []
                self.df = None
                return
            
            # 如果设置了最大样本数，则限制样本数量
            if max_samples is not None and max_samples > 0 and len(self.valid_indices) > max_samples:
                print(f"限制样本数量为 {max_samples} (原有 {len(self.valid_indices)} 条记录)")
                # 随机选择样本，而不是只取前面的
                import random
                random_indices = random.sample(range(len(self.valid_indices)), max_samples)
                self.valid_indices = [self.valid_indices[i] for i in random_indices]
                self.image_paths = [self.image_paths[i] for i in random_indices]
            
            # 保存缓存
            if self.use_cache:
                try:
                    import pickle
                    with open(self.cache_path, 'wb') as f:
                        cache_data = {
                            'df': self.df,
                            'valid_indices': self.valid_indices,
                            'image_paths': self.image_paths
                        }
                        pickle.dump(cache_data, f)
                    print(f"数据集已缓存到: {self.cache_path}")
                except Exception as e:
                    print(f"保存缓存失败: {e}")
        except Exception as e:
            print(f"处理数据失败: {e}")
            # 创建空数据集
            self.valid_indices = []
            self.image_paths = []
            self.df = None
    
    def _load_data_from_file(self, data_path, max_samples=None):
        """
        从文件加载数据
        
        Args:
            data_path: 数据文件路径
            max_samples: 最大样本数
            
        Returns:
            df: DataFrame
            valid_indices: 有效索引
            image_paths: 图像路径
        """
        try:
            print(f"加载数据文件: {data_path}")
            # 加载CSV文件，使用重试机制
            max_retries = 3
            retry_delay = 2  # 秒
            df = None
            
            for retry in range(max_retries):
                try:
                    # 尝试读取CSV文件
                    df = pd.read_csv(
                        data_path,
                        encoding='utf-8',
                        engine='python'
                    )
                    print(f"成功加载数据，共 {len(df)} 条记录")
                    print(f"数据列: {df.columns.tolist()}")
                    break  # 加载成功
                except Exception as e:
                    if retry < max_retries - 1:
                        print(f"第 {retry+1}/{max_retries} 次加载数据失败: {e}，{retry_delay} 秒后重试...")
                        time.sleep(retry_delay)
                    else:
                        print(f"加载数据失败: {e}")
                        return None, [], []
            
            # 如果加载失败
            if df is None or len(df) == 0:
                print("警告：数据为空，返回空结果")
                return None, [], []
            
            # 检查必要的列是否存在
            required_columns = ['path', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"错误：数据文件缺少必要的列: {missing_columns}")
                # 尝试查找可能的替代列
                possible_path_columns = ['path', 'image', 'images', 'img', 'photo', 'image_id', 'image_ids']
                found_path_column = None
                for col in possible_path_columns:
                    if col in df.columns:
                        found_path_column = col
                        print(f"找到可能的图像路径列: {col}")
                        break
                
                if found_path_column:
                    print(f"使用 '{found_path_column}' 列作为图像路径")
                    # 重命名列以匹配预期的列名
                    df = df.rename(columns={found_path_column: 'path'})
                else:
                    print(f"无法找到图像路径列，可用的列: {df.columns.tolist()}")
                    return None, [], []
            
            # 处理数据记录
            valid_indices = []
            image_paths = []
            
            # 批量处理记录以提高效率
            for idx, row in df.iterrows():
                # 获取图像路径
                image_path_str = row['path']
                if pd.isna(image_path_str):
                    continue
                
                # 处理路径（移除可能的前导./）
                image_path_str = image_path_str.strip()
                if image_path_str.startswith('./'):  # 如果路径以./开头，移除它
                    image_path_str = image_path_str[2:]
                
                # 构建完整的图像路径 - 使用绝对路径
                if os.path.isabs(image_path_str):
                    full_image_path = image_path_str
                else:
                    # 使用项目根目录作为基础路径，而不是self.images_dir
                    project_root = '/Users/wujianxiang/Documents/GitHub/models'
                    full_image_path = os.path.join(project_root, image_path_str)
                
                # 检查图像文件是否存在
                if os.path.isfile(full_image_path):
                    valid_indices.append(idx)
                    image_paths.append(full_image_path)
                    if self.verbose:
                        print(f"找到有效图像: {full_image_path}")
                elif self.verbose:
                    print(f"警告：图像文件不存在: {full_image_path}")
            
            print(f"找到 {len(valid_indices)} 条有效记录（具有有效图像路径）")
            
            # 如果没有有效记录，返回空结果
            if not valid_indices:
                print("警告：没有有效的数据记录，返回空结果")
                return None, [], []
            
            return df, valid_indices, image_paths
            
        except Exception as e:
            print(f"处理数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None, [], []
    
    def create_default_image(self, size=(224, 224)):
        """
        创建默认图像（纯黑色）
        
        Args:
            size: 图像大小，默认为(224, 224)
            
        Returns:
            default_img: 默认图像
        """
        # 创建纯黑色图像
        default_img = Image.new('RGB', size, color='black')
        return default_img
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        获取数据集中的一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            sample: 包含图像和标签的样本
        """
        try:
            # 获取有效索引
            valid_idx = self.valid_indices[idx]
            
            # 获取样本数据
            item = self.df.iloc[valid_idx]
            
            # 标签处理 - 确保标签是数字格式
            try:
                if isinstance(item['label'], str):
                    if item['label'].lower() == 'fake':
                        label = 1
                    elif item['label'].lower() == 'real':
                        label = 0
                    else:
                        # 尝试将字符串转换为数字
                        label = int(float(item['label']))
                else:
                    # 如果已经是数字类型，确保是0或1
                    label = int(item['label']) if int(item['label']) in [0, 1] else 0
                    
                if self.verbose:
                    print(f"处理样本 {idx} (原始索引 {valid_idx}): 原始标签={item['label']}, 处理后标签={label}")
            except Exception as e:
                if self.verbose:
                    print(f"标签处理错误: {e}, 使用默认标签0")
                label = 0  # 默认为0（真新闻）
            
            # 图像路径处理
            image_path = self.image_paths[idx]
            
            # 检查图像文件是否存在
            if os.path.isfile(image_path):
                try:
                    # 尝试加载图像
                    image = Image.open(image_path).convert('RGB')
                except (IOError, OSError) as e:
                    if self.verbose:
                        print(f"警告：无法加载图像: {image_path}，错误: {e}，使用默认图像替代")
                    # 使用默认图像替代
                    image = self.create_default_image()
            else:
                if self.verbose:
                    print(f"警告：图像文件不存在: {image_path}，使用默认图像替代")
                # 使用默认图像替代
                image = self.create_default_image()
            
            # 应用数据增强或转换
            if self.transform is not None:
                try:
                    image = self.transform(image)
                except Exception as e:
                    if self.verbose:
                        print(f"警告：图像转换失败: {e}，使用未转换的图像")
            
            # 如果提供了CLIP处理器，则处理图像以获取CLIP输入
            clip_image = None
            if self.clip_processor is not None:
                try:
                    # 确保图像是PIL图像对象
                    if isinstance(image, torch.Tensor):
                        # 如果已经是张量，需要转换回PIL图像
                        # 首先反归一化
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img_unnormalized = image * std + mean
                        # 确保值在[0,1]范围内
                        img_unnormalized = torch.clamp(img_unnormalized, 0, 1)
                        # 转换为PIL图像
                        img_pil = transforms.ToPILImage()(img_unnormalized)
                        clip_inputs = self.clip_processor(images=img_pil, return_tensors="pt")
                    else:
                        # 如果是PIL图像，直接处理
                        clip_inputs = self.clip_processor(images=image, return_tensors="pt")
                    clip_image = clip_inputs.pixel_values.squeeze(0)  # 移除批次维度
                except Exception as e:
                    if self.verbose:
                        print(f"警告：CLIP处理失败: {e}，使用None替代")
            
            # 返回样本字典
            sample = {
                'resnet_image': image,
                'clip_image': clip_image,
                'label': torch.tensor(label, dtype=torch.long)
            }
            
            # 内存管理 - 如果启用了低内存模式，则清理不必要的数据
            if self.low_memory_mode:
                del image
                if clip_image is not None:
                    del clip_image
                gc.collect()
            
            return sample
            
        except Exception as e:
            if self.verbose:
                print(f"错误：获取样本 {idx} 失败: {e}")
            import traceback
            traceback.print_exc()
            
            # 返回默认样本以避免训练中断
            return {
                'resnet_image': self.transform(self.create_default_image()) if self.transform else self.create_default_image(),
                'clip_image': None,
                'label': torch.tensor(0, dtype=torch.long)  # 默认为0（真新闻）
            }

class ImageProcessingModel(nn.Module):
    def __init__(self, num_classes=2, clip_model_path=CLIP_MODEL_PATH, use_local_models=True, fast_mode=False):
        """
        初始化图像处理模型
        
        Args:
            num_classes: 分类类别数
            clip_model_path: CLIP模型路径
            use_local_models: 是否使用本地模型
        """
        super(ImageProcessingModel, self).__init__()
        
        # 加载预训练模型，支持本地模型加载
        local_files_only = use_local_models
        
        # 初始化维度
        self.resnet_dim = 2048  # ResNet-50输出维度
        self.clip_dim = 768  # CLIP视觉编码器输出维度
        
        # 加载ResNet模型（快速模式使用较小的ResNet-18）
        if fast_mode:
            print("快速模式: 加载ResNet-18模型...")
            self.resnet = models.resnet18(pretrained=True)
            self.resnet_dim = 512  # ResNet-18输出维度
        else:
            print("加载ResNet-50模型...")
            self.resnet = models.resnet50(pretrained=True)
            self.resnet_dim = 2048  # ResNet-50输出维度
            
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        print("成功加载ResNet模型")
        
        # 尝试加载CLIP视觉编码器
        try:
            print(f'尝试加载CLIP视觉模型: {clip_model_path}, local_files_only={local_files_only}')
            # 如果是本地路径且存在，直接使用本地路径
            if os.path.exists(clip_model_path):
                self.clip_vision_encoder = CLIPVisionModel.from_pretrained(clip_model_path)
                print(f'成功从本地路径加载CLIP视觉模型: {clip_model_path}')
            else:
                self.clip_vision_encoder = CLIPVisionModel.from_pretrained(clip_model_path, local_files_only=local_files_only)
                print('成功加载CLIP视觉模型')
        except Exception as e:
            print(f'加载CLIP视觉模型失败: {e}')
            print('创建简化的视觉编码器...')
            
            # 如果加载失败，创建一个简化的视觉编码器
            self.clip_vision_encoder = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512, self.clip_dim)
            )
            print('创建了简化的视觉编码器')
        
        # 特征对齐层 - 将ResNet和CLIP特征映射到相同的维度空间
        self.resnet_alignment = nn.Linear(self.resnet_dim, 512)
        self.clip_alignment = nn.Linear(self.clip_dim, 512)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 768),  # 更大的隐藏层
            nn.ReLU(),
            nn.BatchNorm1d(768),
            nn.Dropout(0.5),  # 增加dropout率
            nn.Linear(768, 384),
            nn.ReLU(),
            nn.BatchNorm1d(384),
            nn.Dropout(0.5),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # L2正则化
        self.l2_reg = 5e-5  # 增加L2正则化强度
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # CLIP维度适配器
        self.clip_dim_adapter = None
    
    def forward(self, resnet_image=None, clip_pixel_values=None):
        """
        前向传播
        
        Args:
            resnet_image: ResNet输入图像
            clip_pixel_values: CLIP输入图像像素值
            
        Returns:
            fused_features: 融合后的特征
            resnet_features_aligned: 对齐后的ResNet特征
            clip_features_aligned: 对齐后的CLIP特征
        """
        try:
            # 验证输入
            if resnet_image is None and clip_pixel_values is None:
                raise ValueError("至少需要提供一种图像输入")
            
            # 如果只提供了一种图像，则使用它作为另一种图像的替代
            if resnet_image is None:
                resnet_image = clip_pixel_values
            if clip_pixel_values is None:
                clip_pixel_values = resnet_image
                
            # 检查输入形状
            if not isinstance(resnet_image, torch.Tensor):
                raise TypeError(f"ResNet图像应为Torch张量，实际为{type(resnet_image)}")
            if not isinstance(clip_pixel_values, torch.Tensor):
                raise TypeError(f"CLIP图像应为Torch张量，实际为{type(clip_pixel_values)}")
            
            # 处理ResNet特征
            resnet_features = self.resnet(resnet_image)
            resnet_features = resnet_features.view(resnet_features.size(0), -1)  # 展平
            
            # 处理CLIP特征
            # 判断CLIP模型类型并相应地处理
            if isinstance(self.clip_vision_encoder, nn.Sequential):
                # 简化的视觉编码器
                clip_features = self.clip_vision_encoder(clip_pixel_values)
            else:
                # 预训练CLIP模型
                try:
                    clip_outputs = self.clip_vision_encoder(clip_pixel_values)
                    if hasattr(clip_outputs, 'pooler_output'):
                        clip_features = clip_outputs.pooler_output
                    elif hasattr(clip_outputs, 'last_hidden_state'):
                        # 如果没有pooler_output，使用last_hidden_state的平均值
                        clip_features = clip_outputs.last_hidden_state.mean(dim=1)
                    else:
                        # 如果没有pooler_output和last_hidden_state，使用第一个输出
                        if isinstance(clip_outputs, tuple):
                            clip_features = clip_outputs[0]
                        else:
                            # 如果只有一个输出，则直接使用
                            clip_features = clip_outputs
                    
                    # 检查特征维度是否匹配
                    if clip_features.size(-1) != self.clip_dim:
                        print(f"警告：CLIP特征维度({clip_features.size(-1)})与预期维度({self.clip_dim})不匹配")
                        # 如果维度不匹配，进行调整
                        if clip_features.size(-1) > self.clip_dim:
                            # 如果CLIP特征维度更大，裁剪到预期维度
                            clip_features = clip_features[:, :self.clip_dim]
                        else:
                            # 如果CLIP特征维度更小，填充到预期维度
                            padding = torch.zeros(clip_features.size(0), self.clip_dim - clip_features.size(-1)).to(clip_features.device)
                            clip_features = torch.cat([clip_features, padding], dim=1)
                except Exception as e:
                    print(f"CLIP视觉编码器前向传播失败: {e}")
                    # 创建零特征以避免训练崩溃
                    batch_size = clip_pixel_values.size(0)
                    clip_features = torch.zeros(batch_size, self.clip_dim).to(clip_pixel_values.device)
            
            # CLIP维度适配
            original_dim = clip_features.size(-1)
            print(f"CLIP特征维度: {original_dim}")
            
            # 检查CLIP特征维度是否需要适配
            if clip_features.size(-1) != self.clip_dim:
                print(f"警告：CLIP特征维度({clip_features.size(-1)})与预期维度({self.clip_dim})不匹配")
                
                # 如果维度不匹配，创建适配器
                if not hasattr(self, 'clip_dim_adapter') or self.clip_dim_adapter.in_features != original_dim:
                    print(f"创建维度适配器: {original_dim} -> {self.clip_dim}")
                    self.clip_dim_adapter = nn.Linear(original_dim, self.clip_dim).to(clip_features.device)
                
                # 应用维度适配器
                clip_features = self.clip_dim_adapter(clip_features)
                print(f"适配后的CLIP特征维度: {clip_features.size(-1)}")
            
            # 特征对齐
            resnet_features_aligned = self.resnet_alignment(resnet_features)
            clip_features_aligned = self.clip_alignment(clip_features)
            
            # 特征融合
            combined_features = torch.cat([resnet_features_aligned, clip_features_aligned], dim=1)
            fused_features = self.fusion(combined_features)
            
            return fused_features, resnet_features_aligned, clip_features_aligned
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"图像处理模型前向传播失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 记录输入的形状和类型
            if resnet_image is not None:
                logger.error(f"ResNet图像: 类型={type(resnet_image)}, 形状={resnet_image.shape if hasattr(resnet_image, 'shape') else 'unknown'}")
            if clip_pixel_values is not None:
                logger.error(f"CLIP图像: 类型={type(clip_pixel_values)}, 形状={clip_pixel_values.shape if hasattr(clip_pixel_values, 'shape') else 'unknown'}")
            
            # 创建零特征以避免训练崩溃，使用模型定义的维度
            batch_size = resnet_image.size(0) if resnet_image is not None else clip_pixel_values.size(0) if clip_pixel_values is not None else 1
            device = resnet_image.device if resnet_image is not None else clip_pixel_values.device if clip_pixel_values is not None else torch.device('cpu')
            
            # 使用模型定义的维度创建零特征
            fused_dim = self.fusion_dim if hasattr(self, 'fusion_dim') else self.resnet_dim + self.clip_dim
            fused_features = torch.zeros(batch_size, fused_dim).to(device)
            resnet_features_aligned = torch.zeros(batch_size, self.resnet_dim).to(device)
            clip_features_aligned = torch.zeros(batch_size, self.clip_dim).to(device)
            
            logger.error(f"创建零特征: fused_features={fused_features.shape}, resnet_features_aligned={resnet_features_aligned.shape}, clip_features_aligned={clip_features_aligned.shape}")
            
            return fused_features, resnet_features_aligned, clip_features_aligned

def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=5e-5, model_save_path='fake_news_image_model.pth', gradient_accumulation_steps=1, weight_decay=5e-5, patience=3):
    """
    训练模型
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        device: 设备
        epochs: 训练轮数
        learning_rate: 学习率
        model_save_path: 模型保存路径
        gradient_accumulation_steps: 梯度累积步数
        weight_decay: 权重衰减
        patience: 早停耐心值
        
    Returns:
        model: 训练后的模型
        history: 训练历史
    """
    print("进入训练函数...")
    # 将模型移动到设备
    model = model.to(device)
    
    # 使用AdamW优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 使用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # 早停机制
    best_val_loss = float('inf')
    no_improve_epochs = 0
    best_model_state = None
    
    # 打印数据加载器信息
    print(f"训练数据加载器长度: {len(train_loader)}")
    print(f"验证数据加载器长度: {len(val_loader)}")
    
    # 检查数据加载器是否为空
    if len(train_loader) == 0:
        print("错误: 训练数据加载器为空!")
        return model, history
    
    # 检查第一个批次的内容
    print("检查第一个训练批次...")
    try:
        first_batch = next(iter(train_loader))
        print(f"第一个批次的键: {first_batch.keys()}")
        for key, value in first_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
            else:
                print(f"  {key}: 类型={type(value)}")
    except Exception as e:
        print(f"检查第一个批次时出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 训练循环
    print("开始训练循环...")
    for epoch in range(epochs):
        # 记录开始时间
        epoch_start_time = time.time()
        
        # 训练模式
        model.train()
        train_loss = 0.0
        
        # 创建进度条
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"轮次 {epoch+1}/{epochs} [训练]")
        
        # 重置梯度
        optimizer.zero_grad()
        
        # 批次计数器
        batch_count = 0
        
        # 训练一个轮次
        for i, batch in train_pbar:
            try:
                # 提取数据
                resnet_images = batch['resnet_image'].to(device)
                labels = batch['label'].to(device)
                
                # 打印批次形状信息（仅对前几个批次）
                if epoch == 0 and i < 3:
                    print(f"批次 {i} 形状信息: resnet_images={resnet_images.shape}, labels={labels.shape}")
                
                # 对于CLIP图像，检查是否存在，如果不存在则使用ResNet图像
                if 'clip_image' in batch and batch['clip_image'] is not None:
                    clip_images = batch['clip_image'].to(device)
                    if epoch == 0 and i < 3:
                        print(f"  clip_images={clip_images.shape}")
                else:
                    # 如果没有CLIP图像，使用ResNet图像代替
                    clip_images = resnet_images
                    if epoch == 0 and i < 3:
                        print("  使用resnet_images代替clip_images")
                
                # 前向传播
                outputs, _, _ = model(resnet_image=resnet_images, clip_pixel_values=clip_images)
                
                # 计算损失
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps  # 梯度累积
                
                # 反向传播
                loss.backward()
                
                # 更新进度条
                train_pbar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})
                
                # 累积训练损失
                train_loss += loss.item() * gradient_accumulation_steps
                batch_count += 1
                
                # 梯度累积 - 每n个批次更新一次参数
                if (i + 1) % gradient_accumulation_steps == 0 or (i + 1) == len(train_loader):
                    # 更新参数
                    optimizer.step()
                    optimizer.zero_grad()
                    
                # 定期清理内存
                if i % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
            except Exception as e:
                print(f"训练批次 {i} 出错: {e}")
                # 打印更详细的错误信息
                import traceback
                traceback.print_exc()
                # 打印批次数据的形状和类型
                print(f"批次数据类型: {type(batch)}")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: 形状={value.shape}, 类型={value.dtype}")
                    else:
                        print(f"  {key}: 类型={type(value)}")
                continue
        
        # 计算平均训练损失
        train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        # 验证模式
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        # 创建验证进度条
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"轮次 {epoch+1}/{epochs} [验证]")
        
        # 不计算梯度
        with torch.no_grad():
            for i, batch in val_pbar:
                try:
                    # 提取数据
                    resnet_images = batch['resnet_image'].to(device)
                    labels = batch['label'].to(device)
                    
                    # 对于CLIP图像，检查是否存在，如果不存在则使用ResNet图像
                    if 'clip_image' in batch and batch['clip_image'] is not None:
                        clip_images = batch['clip_image'].to(device)
                    else:
                        # 如果没有CLIP图像，使用ResNet图像代替
                        clip_images = resnet_images
                    
                    # 前向传播
                    outputs, _, _ = model(resnet_image=resnet_images, clip_pixel_values=clip_images)
                    
                    # 计算损失
                    loss = criterion(outputs, labels)
                    
                    # 累积验证损失
                    val_loss += loss.item()
                    
                    # 获取预测结果
                    _, preds = torch.max(outputs, 1)
                    
                    # 收集预测和标签
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"验证批次 {i} 出错: {e}")
                    continue
        
        # 计算平均验证损失
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        # 计算评估指标
        accuracy = accuracy_score(val_labels, val_preds)
        precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
        recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
        f1 = f1_score(val_labels, val_preds, average='binary', zero_division=0)
        
        # 记录训练历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(accuracy)
        history['val_precision'].append(precision)
        history['val_recall'].append(recall)
        history['val_f1'].append(f1)
        
        # 计算轮次时间
        epoch_time = time.time() - epoch_start_time
        
        # 打印轮次结果
        print(f"轮次 {epoch+1}/{epochs} - 时间: {epoch_time:.2f}s - 训练损失: {train_loss:.4f} - 验证损失: {val_loss:.4f} - 准确率: {accuracy:.4f} - 精确度: {precision:.4f} - 召回率: {recall:.4f} - F1: {f1:.4f}")
        
        # 早停检查
        if val_loss < best_val_loss:
            print(f"验证损失改善了 {best_val_loss - val_loss:.6f}")
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve_epochs = 0
            
            # 保存最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f"模型已保存到 {model_save_path}")
        else:
            no_improve_epochs += 1
            print(f"验证损失没有改善，连续 {no_improve_epochs} 轮次")
            
            # 如果连续多个轮次没有改善，则提前停止训练
            if no_improve_epochs >= patience:
                print(f"早停: 验证损失在 {patience} 轮次内没有改善")
                break
        
        # 清理内存
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    # 加载最佳模型状态
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("已加载最佳模型状态")
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    评估模型
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        
    Returns:
        metrics: 评估指标
    """
    print("开始评估模型...")
    model.eval()
    all_preds = []
    all_labels = []
    
    # 使用tqdm显示进度
    with torch.no_grad():
        test_iterator = tqdm(test_loader, desc='评估模型')
        for batch in test_iterator:
            # 获取数据并移动到目标设备
            resnet_images = batch['resnet_image'].to(device)
            labels = batch['label'].to(device)
            
            # 对于CLIP图像，检查是否存在，如果不存在则使用ResNet图像
            if 'clip_image' in batch and batch['clip_image'] is not None:
                clip_images = batch['clip_image'].to(device)
            else:
                # 如果没有CLIP图像，使用ResNet图像代替
                clip_images = resnet_images
            
            # 前向传播
            fused_features, _, _ = model(resnet_image=resnet_images, clip_pixel_values=clip_images)
            preds = torch.argmax(fused_features, dim=1).cpu().numpy()
            
            # 收集预测结果和标签
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # 释放内存
            del resnet_images, clip_images, fused_features
    
    # 计算评估指标
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # 打印评估结果
    print(f'测试准确率: {acc:.4f}')
    print(f'测试精确率: {precision:.4f}')
    print(f'测试召回率: {recall:.4f}')
    print(f'测试F1分数: {f1:.4f}')
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 绘制混淆矩阵
    try:
        plt.figure(figsize=(8, 6), dpi=100)  # 减小图形尺寸和DPI以减少内存使用
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_image.png')
        plt.close()
        print("混淆矩阵已保存到 confusion_matrix_image.png")
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")
    
    # 返回评估指标
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }
    
    return metrics

def plot_training_history(history):
    """
    绘制训练历史
    
    Args:
        history: 训练历史
    """
    # 创建一个2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)  # 减小DPI以减少内存使用
    
    # 绘制训练损失
    axes[0, 0].plot(history['train_loss'])
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    
    # 绘制验证损失
    axes[0, 1].plot(history['val_loss'])
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    
    # 绘制验证准确率
    axes[1, 0].plot(history['val_accuracy'])
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    
    # 绘制验证F1值
    axes[1, 1].plot(history['val_f1'])
    axes[1, 1].set_title('Validation F1 Score')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    try:
        plt.savefig('training_history.png')
        print('训练历史图形已保存到 training_history.png')
    except Exception as e:
        print(f'保存训练历史图形时出错: {e}')
    
    plt.close()

def setup_device():
    """
    设置设备并处理可能的兼容性问题，优先使用CUDA（适用于RTX 3090）
    
    Returns:
        device: 可用的设备（cuda或cpu）
    """
    # 检查CUDA是否可用
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用CUDA设备加速: {torch.cuda.get_device_name(0)}")
        # 打印CUDA设备信息
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备属性: {torch.cuda.get_device_properties(0)}")
        return device
    
    # CUDA不可用的情况
    print("CUDA不可用，请检查GPU驱动和CUDA安装")
    
    # 使用CPU
    device = torch.device('cpu')
    print("使用CPU设备")
    return device

def main(use_small_dataset=False, fast_mode=False, args=None):
    """
    主函数
    
    Args:
        use_small_dataset: 是否使用小数据集进行测试
        fast_mode: 是否使用快速训练模式
        args: 命令行参数
    """
    print("程序开始执行...")
    # 设置随机种子以确保可重复性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 忽略警告
    warnings.filterwarnings('ignore')
    
    # 设置超时处理
    def timeout_handler(signum, frame):
        raise TimeoutError("操作超时")
    
    # 设置设备
    device = setup_device()
    print(f"使用设备: {device}")
    
    # 在开始前先清理内存
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 设置数据路径
    data_dir = '/Users/wujianxiang/Documents/GitHub/models/data'
    
    # 根据是否使用小数据集选择不同的数据文件
    if use_small_dataset:
        train_file = os.path.join(data_dir, 'small_train.csv')
        val_file = os.path.join(data_dir, 'small_val.csv')  # 使用专门的验证集文件
        test_file = os.path.join(data_dir, 'small_test.csv')
    else:
        train_file = os.path.join(data_dir, 'train.csv')
        val_file = os.path.join(data_dir, 'val.csv')  # 使用专门的验证集文件
        test_file = os.path.join(data_dir, 'test.csv')
    
    # 图像目录现在是相对于数据目录的
    train_images_dir = os.path.join('/Users/wujianxiang/Documents/GitHub/models')
    test_images_dir = os.path.join('/Users/wujianxiang/Documents/GitHub/models')
    
    # 检查数据文件是否存在
    for file_path in [train_file, val_file, test_file]:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在: {file_path}")
    
    if not os.path.exists(train_images_dir):
        print(f"错误: 训练图像目录不存在: {train_images_dir}")
        return
    
    if not os.path.exists(test_images_dir):
        print(f"警告: 测试图像目录不存在: {test_images_dir}")
    
    # 设置图像尺寸 - 快速模式使用更小的图像
    if fast_mode:
        image_size = (224, 224)  # 保持与CLIP模型兼容的尺寸
    else:
        image_size = (224, 224)  # 保持与CLIP模型兼容的尺寸
    
    # 设置最大样本数 - 用于控制内存使用
    max_samples = MAX_SAMPLES  # 默认限制样本数为8000
    if args and hasattr(args, 'max_samples') and args.max_samples is not None and args.max_samples > 0:
        max_samples = args.max_samples
    print(f"限制每个数据集的最大样本数为: {max_samples}")
    
    # 设置是否使用缓存
    use_cache = True
    if args and hasattr(args, 'no_cache') and args.no_cache:
        use_cache = False
        print("禁用数据集缓存")
    
    # 加载CLIP处理器
    print("加载CLIP处理器...")
    try:
        clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        print("成功加载CLIP处理器")
    except Exception as e:
        print(f"加载CLIP处理器失败: {e}，将使用替代方法")
        clip_processor = None
    
    # 创建数据集，使用内存优化选项
    print("创建训练数据集...")
    train_dataset = FakeNewsImageDataset(
        data_path=train_file, 
        images_dir=train_images_dir, 
        clip_processor=clip_processor,
        use_cache=use_cache,
        max_samples=max_samples,
        image_size=image_size,
        low_memory_mode=args.low_memory if args else False,  # 启用低内存模式
        data_augmentation=True,
        verbose=False
    )
    
    # 如果使用小数据集，则分割训练集为训练集和验证集
    if use_small_dataset:
        small_size = min(50, len(train_dataset))  # 限制小数据集的大小
        indices = list(range(small_size))
        train_dataset = Subset(train_dataset, indices)
        print(f"小数据集大小: {len(train_dataset)}")
        
        # 分割数据集为训练集和验证集
        total_size = len(train_dataset)
        # 确保分割比例与原始设计相同，但适应实际数据集大小
        train_ratio = 0.8  # 80%用于训练，20%用于验证
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        print(f"使用小数据集模式 - 分割为训练集({train_size}条)和验证集({val_size}条)")
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        print(f"训练集大小: {len(train_dataset)}")
        # 创建验证数据集 - 使用训练集的一小部分作为验证集
        total_size = len(train_dataset)
        # 确保分割比例与原始设计相同，但适应实际数据集大小
        train_ratio = 0.8  # 80%用于训练，20%用于验证
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size
        
        print(f"使用小数据集模式 - 分割为训练集({train_size}条)和验证集({val_size}条)")
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    print(f"训练集大小: {len(train_dataset)}")
    
    # 尝试加载测试集，如果失败则使用验证集的一部分
    print("创建测试数据集...")
    try:
        test_dataset = FakeNewsImageDataset(
            data_path=test_file, 
            images_dir=test_images_dir, 
            clip_processor=clip_processor,
            use_cache=use_cache,
            max_samples=max_samples,
            image_size=image_size,
            low_memory_mode=args.low_memory if args else False,  # 启用低内存模式
            data_augmentation=False,
            verbose=False
        )
        
        # 如果测试集为空，使用验证集的一部分作为测试集
        if len(test_dataset) == 0:
            print("测试集为空，使用验证集的一部分作为测试集")
            test_size = min(5, len(val_dataset))
            test_indices = list(range(test_size))
            test_dataset = Subset(val_dataset, test_indices)
            print(f"使用验证集的前 {test_size} 条记录作为测试集")
        
        print(f"测试集大小: {len(test_dataset)}")
    except Exception as e:
        print(f"加载测试集失败: {e}，将使用验证集的一部分作为测试集")
        # 如果测试集加载失败，使用验证集的一部分作为测试集
        test_size = min(5, len(val_dataset))
        test_indices = list(range(test_size))
        test_dataset = Subset(val_dataset, test_indices)
        print(f"使用验证集的前 {test_size} 条记录作为测试集")
    
    # 设置批量大小和工作线程数 - 减小批量大小以减少内存使用
    batch_size = args.batch_size if args else CUDA_BATCH_SIZE
    num_workers = args.workers if args else CUDA_NUM_WORKERS  # 使用命令行参数覆盖工作线程数（如果提供）
    prefetch_factor = 2
    
    print(f"批量大小: {batch_size}, 工作线程数: {num_workers}, 预取因子: {prefetch_factor}")
    
    # 创建数据加载器，使用更小的批量大小和更少的工作线程
    dataloader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if torch.cuda.is_available() else False,
        'prefetch_factor': prefetch_factor if num_workers > 0 else None,
        'persistent_workers': False  # 禁用持久工作线程以减少内存使用
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **dataloader_kwargs)
    
    # 创建模型
    print("创建模型...")
    model = ImageProcessingModel(num_classes=2, use_local_models=True, fast_mode=fast_mode)
    model = model.to(device)
    print("模型创建完成")
    
    # 设置训练参数
    if fast_mode:
        epochs = args.epochs if args and hasattr(args, 'epochs') and args.epochs > 0 else 1  # 快速模式只训练1轮
        learning_rate = args.learning_rate if args and hasattr(args, 'learning_rate') and args.learning_rate > 0 else 2e-4
        gradient_accumulation_steps = args.gradient_accumulation if args and hasattr(args, 'gradient_accumulation') and args.gradient_accumulation > 0 else 4  # 增加梯度累积步数以使用更小的批量大小
    else:
        epochs = args.epochs if args and hasattr(args, 'epochs') and args.epochs > 0 else 5  # 减少训练轮数以加快训练速度
        learning_rate = args.learning_rate if args and hasattr(args, 'learning_rate') and args.learning_rate > 0 else 2e-4
        gradient_accumulation_steps = args.gradient_accumulation if args and hasattr(args, 'gradient_accumulation') and args.gradient_accumulation > 0 else 16  # 使用更大的梯度累积步数
    
    print(f"训练轮数: {epochs}, 学习率: {learning_rate}, 梯度累积步数: {gradient_accumulation_steps}")
    
    # 创建模型保存目录
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_save_path = os.path.join(model_dir, 'fake_news_image_model.pth')
    
    # 训练模型
    print("开始训练模型...")
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        model_save_path=model_save_path,
        gradient_accumulation_steps=gradient_accumulation_steps,
        weight_decay=args.weight_decay if args and hasattr(args, 'weight_decay') else 1e-5,
        patience=args.patience if args and hasattr(args, 'patience') else 5
    )
    
    # 绘制训练历史
    print("绘制训练历史...")
    plot_training_history(history)
    
    # 评估模型
    print("评估模型...")
    metrics = evaluate_model(model, test_loader, device)
    
    # 打印最终评估结果，添加更美观的格式
    print("\n" + "="*50)
    print("最终评估结果:")
    print("-"*50)
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1值:   {metrics['f1']:.4f}")
    print("="*50)
    
    print("模型测试完成")
    return metrics

if __name__ == '__main__':
    # 添加命令行参数支持
    import argparse
    print("解析命令行参数...")
    parser = argparse.ArgumentParser(description='多模态假新闻检测模型')
    parser.add_argument('--small', action='store_true', help='使用小数据集进行测试')
    parser.add_argument('--no-cache', action='store_true', help='不使用缓存加载数据集')
    parser.add_argument('--fast', action='store_true', help='使用快速训练模式（简化模型）')
    parser.add_argument('--max-samples', type=int, default=None, help='限制每个数据集的最大样本数')
    parser.add_argument('--batch-size', type=int, default=CUDA_BATCH_SIZE, help='批量大小')
    parser.add_argument('--epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--low-memory', action='store_true', help='启用低内存模式')
    parser.add_argument('--gradient-accumulation', type=int, default=4, help='梯度累积步数')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='学习率')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    parser.add_argument('--train-size', type=int, default=None, help='训练集大小')
    parser.add_argument('--test-size', type=int, default=None, help='测试集大小')
    parser.add_argument('--workers', type=int, default=CUDA_NUM_WORKERS, help='数据加载器工作线程数')
    parser.add_argument('--gpu', type=int, default=0, help='使用的GPU编号')
    args = parser.parse_args()
    print("解析命令行参数完成")
    
    # 根据命令行参数决定是否使用小数据集和快速模式
    use_small = args.small
    fast_mode = args.fast
    
    if use_small:
        print("使用小数据集模式运行")
    else:
        print("使用完整数据集模式运行")
        
    if fast_mode:
        print("启用快速训练模式（简化模型）")
    
    # 设置环境变量以优化性能
    os.environ['OMP_NUM_THREADS'] = '4'  # 限制OpenMP线程数
    
    print("环境变量设置完成")
    
    try:
        print("开始调用main函数...")
        main(use_small_dataset=use_small, fast_mode=fast_mode, args=args)
        print("main函数执行完成")
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        print("清理资源...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("程序已安全退出")
