# data_loader.py

import os
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# 定义支持的图像格式
SUPPORTED_IMAGE_TYPES = ['.jpg', '.jpeg', '.png', '.webp', '.tif', '.tiff']

def _load_image(path: str) -> np.ndarray:
    """
    一个健壮的图像加载函数，支持标准格式和16位TIFF。
    所有图像都被转换为 [0, 1] 范围的 float32 NumPy 数组，并确保是3通道RGB。
    """
    # 获取文件扩展名
    ext = os.path.splitext(path)[1].lower()

    if ext in ['.tif', '.tiff']:
        # 使用OpenCV读取TIFF，IMREAD_UNCHANGED可以保留原始位深度
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"无法使用OpenCV加载图像: {path}")

        # 检查是否为16位图像，并进行归一化到 [0, 1]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
        # 检查是否为8位图像，并进行归一化到 [0, 1]
        elif img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        
        # 如果是单通道（灰度图），则复制成3个通道
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        # 如果有4个通道（如RGBA），则只取RGB通道
        elif img.shape[2] == 4:
            img = img[:, :, :3]
            
        # OpenCV读取的顺序是BGR，转换为RGB
        # 注意: 如果原始tif是灰度的，cvtColor会报错，但我们已经手动堆叠成3通道，所以是安全的
        if img.shape[2] == 3:
             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    else:
        # 使用Pillow处理其他标准图像格式
        with Image.open(path) as img_pil:
            img_pil = img_pil.convert("RGB")
            img = np.array(img_pil, dtype=np.float32) / 255.0
            
    return img

class PairedImageDataset(Dataset):
    """
    用于加载成对 LQ/HQ 图像的新数据集类。
    通过一个包含 lq_path 和 hq_path 的数据源列表进行初始化。
    """
    def __init__(self, data_sources: list, split: str, use_dynamic_lq: bool, resolution=512):
        """
        初始化数据集。
        :param data_sources: 一个字典列表，每个字典包含 'lq_path' 和 'hq_path'。
        :param split: 要使用的数据子目录，例如 'train', 'validate', 'test'。
        :param use_dynamic_lq: 是否使用所有匹配的LQ文件 (xxx_yyy.ext)。
        :param resolution: 图像输出分辨率。
        """
        self.image_pairs = []
        self.resolution = resolution
        
        # 定义图像变换流程，将加载的图像转换为模型所需的格式
        self.transform = transforms.Compose([
            transforms.ToTensor(), # 将 [H, W, C] 的 NumPy 数组(0-1范围)转换为 [C, H, W] 的 Tensor(0-1范围)
            transforms.Resize((resolution, resolution), antialias=True), # 调整到指定分辨率
            transforms.Normalize([0.5], [0.5]) # 归一化到 [-1, 1]，这是VAE的标准输入范围
        ])

        print(f"正在为 '{split}' 集扫描数据源并匹配 LQ/HQ 图像对...")
        
        # 遍历YAML中提供的每个数据源对
        for source in data_sources:
            lq_dir = source['lq_path']
            hq_dir = source['hq_path']
            
            # 根据 'split' 参数定位到具体的 'train' 或 'validate' 目录
            lq_split_path = os.path.join(lq_dir, split)
            hq_split_path = os.path.join(hq_dir, split)

            if not os.path.isdir(lq_split_path) or not os.path.isdir(hq_split_path):
                print(f"警告: 目录 {lq_split_path} 或 {hq_split_path} 不存在，已跳过。")
                continue

            # 遍历LQ目录下的所有文件
            for lq_filename in os.listdir(lq_split_path):
                lq_filepath = os.path.join(lq_split_path, lq_filename)
                
                if not os.path.isfile(lq_filepath) or not any(lq_filename.lower().endswith(ext) for ext in SUPPORTED_IMAGE_TYPES):
                    continue

                base_name, ext = os.path.splitext(lq_filename)
                parts = base_name.rsplit('_', 1)
                
                if len(parts) == 2 and parts[1].isdigit():
                    hq_base_name = parts[0]
                    lq_suffix = parts[1]
                    
                    if not use_dynamic_lq and lq_suffix != '001':
                        continue
                else:
                    continue

                # 寻找对应的HQ文件
                found_hq = False
                for hq_ext in SUPPORTED_IMAGE_TYPES:
                    hq_filepath = os.path.join(hq_split_path, f"{hq_base_name}{hq_ext}")
                    if os.path.isfile(hq_filepath):
                        self.image_pairs.append((lq_filepath, hq_filepath))
                        found_hq = True
                        break
                
                if not found_hq:
                    # print(f"调试: 找不到 {lq_filepath} 对应的 HQ 文件，已跳过。") # 可选的调试信息
                    pass

        print(f"'{split}' 集数据扫描完成，共找到 {len(self.image_pairs)} 个训练图像对。")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        lq_path, hq_path = self.image_pairs[idx]
        
        try:
            lq_image_np = _load_image(lq_path)
            hq_image_np = _load_image(hq_path)
        except Exception as e:
            print(f"加载图像时出错: {e}. 文件路径: {lq_path}, {hq_path}")
            # 返回一个占位符或跳过，这里我们选择用第一张图代替
            lq_path, hq_path = self.image_pairs[0]
            lq_image_np = _load_image(lq_path)
            hq_image_np = _load_image(hq_path)

        lq_image = self.transform(lq_image_np)
        hq_image = self.transform(hq_image_np)

        return {"lq_image": lq_image, "hq_image": hq_image}