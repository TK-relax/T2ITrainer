import torch
import yaml
import importlib
import numpy as np
import matplotlib.pyplot as plt

# 确保可以从 data_loader.py 导入
from data_loader import PairedImageDataset

CONFIG_YAML_PATH = "config.yaml"
# 可视化时，每行显示的图片数量
IMAGES_PER_ROW = 4

def initialize_dataloaders_from_yaml(config_path, batch_size, num_workers=10):
    """
    从YAML配置文件中读取设置，并初始化训练和验证的数据加载器。
    (这个函数与主训练脚本中的版本相同)
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'paired_dataset' not in config:
        raise ValueError("YAML文件中缺少 'paired_dataset' 顶级键。")

    data_config = config['paired_dataset']
    
    dataloaders = {}
    
    for split in ['train', 'validate']:
        if split in data_config:
            print(f"正在初始化 '{split}' 数据加载器...")
            
            dataset_config = data_config[split]
            module_path, class_name = dataset_config['target'].rsplit('.', 1)
            module = importlib.import_module(module_path)
            DatasetClass = getattr(module, class_name)
            
            # 实例化数据集时，手动传入 resolution
            params = dataset_config['params']
            params['resolution'] = 512 # 在测试中硬编码或从别处读取
            
            dataset = DatasetClass(**params)
            
            if len(dataset) == 0:
                print(f"警告: '{split}' 数据集为空，请检查路径和文件命名。")
                dataloaders[split] = None
                continue

            is_train = split == 'train'
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=min(batch_size, len(dataset)), # 防止batch_size大于数据集大小
                shuffle=is_train,
                num_workers=num_workers
            )
            dataloaders[split] = dataloader

    return dataloaders.get('train'), dataloaders.get('validate')


def visualize_batch(batch, title, output_filename):
    """
    可视化一个批次中的 LQ/HQ 图像对。
    """
    lq_images = batch['lq_image']
    hq_images = batch['hq_image']
    batch_size = lq_images.shape[0]

    # 创建一个能容纳所有图像对的图表
    fig, axes = plt.subplots(2, batch_size, figsize=(batch_size * 4, 8))
    fig.suptitle(title, fontsize=16)

    for i in range(batch_size):
        # --- 处理LQ图像 ---
        lq_img_tensor = lq_images[i]
        # 反归一化：从 [-1, 1] 转换回 [0, 1]
        lq_img_tensor = lq_img_tensor * 0.5 + 0.5
        # 将张量从 [C, H, W] 转换为 [H, W, C] 以便 matplotlib 显示
        lq_img_np = lq_img_tensor.permute(1, 2, 0).cpu().numpy()
        # clip确保数值在[0, 1]范围内，防止浮点误差
        lq_img_np = np.clip(lq_img_np, 0, 1)

        ax_lq = axes[0, i] if batch_size > 1 else axes[0]
        ax_lq.imshow(lq_img_np)
        ax_lq.set_title(f"LQ Image {i+1}")
        ax_lq.axis('off')

        # --- 处理HQ图像 ---
        hq_img_tensor = hq_images[i]
        hq_img_tensor = hq_img_tensor * 0.5 + 0.5
        hq_img_np = hq_img_tensor.permute(1, 2, 0).cpu().numpy()
        hq_img_np = np.clip(hq_img_np, 0, 1)

        ax_hq = axes[1, i] if batch_size > 1 else axes[1]
        ax_hq.imshow(hq_img_np)
        ax_hq.set_title(f"HQ Image {i+1}")
        ax_hq.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 将 plt.show() 替换为 plt.savefig()
    # plt.show() 
    plt.savefig(output_filename)
    print(f"可视化结果已保存到: {output_filename}")
    
    # 关闭图像，释放内存
    plt.close(fig)


if __name__ == "__main__":
    print("开始测试 DataLoader...")
    
    # 初始化 DataLoader
    train_loader, val_loader = initialize_dataloaders_from_yaml(
        config_path=CONFIG_YAML_PATH,
        batch_size=IMAGES_PER_ROW
    )

     # --- 测试训练数据加载器 ---
    if train_loader:
        print("\n从训练数据加载器中获取一个批次...")
        try:
            train_batch = next(iter(train_loader))
            print(f"成功获取训练批次，包含 {len(train_batch['lq_image'])} 张图片。")
            # 调用时传入一个文件名
            visualize_batch(train_batch, "Training Data Batch Sample", "train_batch_visualization.png")
        except StopIteration:
            print("错误：训练数据加载器为空，无法获取数据。")
    else:
        print("训练数据加载器未成功初始化，跳过测试。")

    # --- 测试验证数据加载器 ---
    if val_loader:
        print("\n从验证数据加载器中获取一个批次...")
        try:
            val_batch = next(iter(val_loader))
            print(f"成功获取验证批次，包含 {len(val_batch['lq_image'])} 张图片。")
            # 调用时传入另一个文件名
            visualize_batch(val_batch, "Validation Data Batch Sample", "val_batch_visualization.png")
        except StopIteration:
            print("错误：验证数据加载器为空，无法获取数据。")
    else:
        print("验证数据加载器未成功初始化，跳过测试。")

    print("\nDataLoader 测试结束。")