# data_loader.py

import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import random

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
    实现了按数据源进行固定且均衡的抽样功能。
    """
    # 1. 修改 __init__ 的函数签名，使用新的参数名
    def __init__(self, data_sources: list, split: str, use_dynamic_lq: bool, resolution=512, config_dir: str = '.', 
                 validate_samples_per_source=None, validation_seed=42):
        
        self.image_pairs = []
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((resolution, resolution), antialias=True),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"--- 正在初始化 '{split}' 数据集 ---")
        
        # 2. 修改数据加载和抽样逻辑
        # 遍历YAML中提供的每个数据源对
        for source_idx, source in enumerate(data_sources):
            lq_dir = source['lq_path']
            hq_dir = source['hq_path']
            
            lq_split_path = os.path.join(config_dir, lq_dir, split)
            hq_split_path = os.path.join(config_dir, hq_dir, split)

            # 用于存储当前数据源找到的所有图像对
            source_specific_pairs = []
            
            if not os.path.isdir(lq_split_path) or not os.path.isdir(hq_split_path):
                print(f"  - 警告: 跳过数据源 {lq_dir} -> {hq_dir}，因为目录不存在。")
                continue

            # (这段文件扫描逻辑与之前相同)
            for lq_filename in os.listdir(lq_split_path):
                # ... (匹配和查找文件的逻辑不变) ...
                # ... 当找到一对时，将其添加到 source_specific_pairs 中 ...
                base_name, ext = os.path.splitext(lq_filename)
                parts = base_name.rsplit('_', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    hq_base_name = parts[0]
                    if not use_dynamic_lq and parts[1] != '001':
                        continue
                else:
                    continue

                found_hq = False
                for hq_ext in SUPPORTED_IMAGE_TYPES:
                    hq_filepath = os.path.join(hq_split_path, f"{hq_base_name}{hq_ext}")
                    if os.path.isfile(hq_filepath):
                        source_specific_pairs.append((os.path.join(lq_split_path, lq_filename), hq_filepath))
                        found_hq = True
                        break
            
            original_source_count = len(source_specific_pairs)
            
            # 3. 对当前数据源的列表进行处理（抽样或全部添加）
            if split == 'validate' and validate_samples_per_source is not None and validate_samples_per_source > 0:
                # 固定抽样逻辑
                if original_source_count == 0:
                    print(f"  - 数据源 '{lq_dir}': 未找到任何图像对。")
                    continue

                # 设置随机种子以保证可复现性
                # 我们可以在种子中加入 source_idx，确保每个源的打乱模式不同，但仍然是固定的
                random.seed(validation_seed + source_idx)
                # 原地打乱列表
                random.shuffle(source_specific_pairs)
                
                # 取前 N 个样本
                num_to_sample = min(validate_samples_per_source, original_source_count)
                sampled_pairs = source_specific_pairs[:num_to_sample]
                
                print(f"  - 数据源 '{lq_dir}': 找到 {original_source_count} 对, 已固定抽样 {len(sampled_pairs)} 对。")
                self.image_pairs.extend(sampled_pairs)
            else:
                # 对于训练集或不进行抽样的验证集，直接全部添加
                print(f"  - 数据源 '{lq_dir}': 找到并添加 {original_source_count} 对。")
                self.image_pairs.extend(source_specific_pairs)

        print(f"--- '{split}' 数据集初始化完成，总共加载了 {len(self.image_pairs)} 个图像对。 ---\n")

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