# test_dataloader.py
# 用于测试 data_loader.py 和 config.yaml 配置是否正常工作的完整脚本。
# 功能：
# 1. 从 config.yaml 加载数据集配置。
# 2. 初始化训练和验证 DataLoader。
# 3. 打印加载到的样本总数，以验证抽样逻辑。
# 4. 可视化一个批次的 LQ/HQ 图像对并保存为文件，供人工检查。

import torch
import yaml
import importlib
import numpy as np
import matplotlib.pyplot as plt
import os

# 确保可以从同一目录下的 data_loader.py 导入 PairedImageDataset 类
# 如果 vscode 提示找不到模块，可以忽略，直接从终端运行即可
from data_loader import PairedImageDataset

# --- 可配置参数 ---
CONFIG_YAML_PATH = "config.yaml"
# 可视化时，一个批次加载多少张图片（建议 2 到 4 张以便观察）
IMAGES_TO_VISUALIZE = 4 

def initialize_dataloaders_from_yaml(config_path, batch_size, num_workers=0):
    """
    从YAML配置文件中读取设置，并初始化训练和验证的数据加载器。
    此函数设计得足够灵活，可以自动处理 params 中的新参数，如 
    validate_samples_per_source 和 validation_seed。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}。请确保它与本脚本在同一目录下。")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'paired_dataset' not in config:
        raise ValueError("YAML文件中缺少 'paired_dataset' 顶级键。")
        
    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_config = config['paired_dataset']
    dataloaders = {}
    
    for split in ['train', 'validate']:
        if split in data_config:
            dataset_config = data_config[split]
            print(f"--- 正在准备 '{split}' 数据加载器 ---")
            
            # 动态导入数据集类，例如 'data_loader.PairedImageDataset'
            module_path, class_name = dataset_config['target'].rsplit('.', 1)
            try:
                module = importlib.import_module(module_path)
                DatasetClass = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                print(f"错误：无法从 '{dataset_config['target']}' 导入类。请检查 target 路径是否正确。")
                raise e

            # 准备传递给数据集构造函数的参数
            params = dataset_config.get('params', {})
            params['resolution'] = params.get('resolution', 1024) # 如果YAML中没定义，则使用默认值
            params['config_dir'] = config_dir
            
            # 实例化数据集
            dataset = DatasetClass(**params)
            
            if len(dataset) == 0:
                print(f"!!! 警告: '{split}' 数据集为空，数据加载器将不会被创建。请检查您的数据路径和文件命名。")
                dataloaders[split] = None
                continue

            # 创建数据加载器
            is_train = split == 'train'
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(batch_size, len(dataset)),
                shuffle=is_train,
                num_workers=num_workers
            )
            dataloaders[split] = dataloader
            print(f"--- '{split}' 数据加载器准备就绪 ---")

    return dataloaders.get('train'), dataloaders.get('validate')


def visualize_batch(batch, title, output_filename):
    """
    可视化一个批次中的 LQ/HQ 图像对，并将其保存到文件。
    """
    if not batch:
        print(f"无法可视化 '{title}'，因为批次数据为空。")
        return

    lq_images = batch['lq_image']
    hq_images = batch['hq_image']
    batch_size = lq_images.shape[0]

    # 创建一个能容纳所有图像对的图表
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8.5))
    fig.suptitle(title, fontsize=16)

    # 如果只有一个图像，axes 不是数组，需要特殊处理
    if batch_size == 1:
        axes = np.array([axes]).T

    for i in range(batch_size):
        # --- 处理LQ图像 ---
        lq_img_tensor = lq_images[i]
        # 反归一化：从 [-1, 1] 转换回 [0, 1]
        lq_img_tensor = lq_img_tensor * 0.5 + 0.5
        # 将张量从 [C, H, W] 转换为 [H, W, C] 以便 matplotlib 显示
        lq_img_np = lq_img_tensor.permute(1, 2, 0).cpu().numpy()
        lq_img_np = np.clip(lq_img_np, 0, 1)

        axes[0, i].imshow(lq_img_np)
        axes[0, i].set_title(f"LQ Image {i+1}")
        axes[0, i].axis('off')

        # --- 处理HQ图像 ---
        hq_img_tensor = hq_images[i]
        hq_img_tensor = hq_img_tensor * 0.5 + 0.5
        hq_img_np = hq_img_tensor.permute(1, 2, 0).cpu().numpy()
        hq_img_np = np.clip(hq_img_np, 0, 1)

        axes[1, i].imshow(hq_img_np)
        axes[1, i].set_title(f"HQ Image {i+1}")
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_filename)
    print(f"✅ 可视化结果已保存到: {output_filename}")
    plt.close(fig)


if __name__ == "__main__":
    print("========= 开始 DataLoader 完整性与抽样逻辑测试 =========\n")
    
    # 1. 初始化 DataLoader
    train_loader, val_loader = initialize_dataloaders_from_yaml(
        config_path=CONFIG_YAML_PATH,
        batch_size=IMAGES_TO_VISUALIZE
    )

    # 2. 验证和报告数据集大小
    print("\n========= 数据集大小验证 =========")
    if train_loader:
        train_size = len(train_loader.dataset)
        print(f"✅ 训练集加载成功，总样本数: {train_size}")
    else:
        print("❌ 训练集加载失败或为空。")

    if val_loader:
        val_size = len(val_loader.dataset)
        print(f"✅ 验证集加载成功，总样本数: {val_size}")
        print("   (请检查此数量是否与您在YAML中配置的'validate_samples_per_source'和数据源数量相符)")
    else:
        print("❌ 验证集加载失败或为空。")

    # 3. 可视化一个批次的数据
    print("\n========= 数据批次可视化测试 =========")
    # --- 测试训练数据加载器 ---
    if train_loader:
        print("正在从训练集获取一个批次用于可视化...")
        train_batch = next(iter(train_loader))
        visualize_batch(train_batch, "Training Data Batch Sample", "test_train_batch.png")
    else:
        print("跳过训练集可视化，因为加载器为空。")

    # --- 测试验证数据加载器 ---
    if val_loader:
        print("\n正在从(抽样后的)验证集获取一个批次用于可视化...")
        val_batch = next(iter(val_loader))
        visualize_batch(val_batch, "Validation Data (Sampled) Batch", "test_validation_batch.png")
    else:
        print("跳过验证集可视化，因为加载器为空。")

    print("\n========= DataLoader 测试脚本执行完毕 =========\n")