"""
数据加载与预处理模块
处理CIFAR-10数据集的下载、加载、预处理和可视化
"""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from config import *


def get_data_loaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """
    加载CIFAR-10数据集并创建数据加载器
    
    数据预处理流程：
    训练集：随机裁剪 -> 水平翻转 -> 转张量 -> 归一化
    测试集：转张量 -> 归一化
    """
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # 随机裁剪，增加数据多样性
        transforms.RandomHorizontalFlip(),      # 随机水平翻转，数据增强
        transforms.ToTensor(),                  # 转为张量，归一化到[0,1]
        transforms.Normalize((0.4914, 0.4822, 0.4465),  # CIFAR-10均值
                             (0.2470, 0.2435, 0.2616))  # CIFAR-10标准差
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    # 加载训练集
    train_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=True,
        download=True,
        transform=transform_train
    )

    # 加载测试集
    test_dataset = datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform_test
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,           # 训练时打乱数据
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader, train_dataset, test_dataset


def visualize_dataset_samples(dataset, save_path=None):
    """
    可视化数据集中的样本图片
    展示每个类别的代表性图像
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    # 为每个类别找一张样本
    class_samples = {i: None for i in range(10)}
    for img, label in dataset:
        if class_samples[label] is None:
            class_samples[label] = img
        if all(v is not None for v in class_samples.values()):
            break

    # 反归一化以便显示
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for i in range(10):
        img = class_samples[i]
        img = img * std + mean  # 反归一化
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()  # CHW -> HWC

        axes[i].imshow(img)
        axes[i].set_title(CLASS_NAMES[i], fontsize=12)
        axes[i].axis('off')

    plt.suptitle('CIFAR-10 Dataset Samples (One per Class)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"数据集样本图已保存至: {save_path}")
    plt.show()
    plt.close()


def print_dataset_info(train_dataset, test_dataset):
    """打印数据集基本信息"""
    print("=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"训练集样本数: {len(train_dataset):,}")
    print(f"测试集样本数: {len(test_dataset):,}")
    print(f"图像尺寸: 32 x 32 x 3 (RGB)")
    print(f"类别数: {NUM_CLASSES}")
    print(f"类别名称: {CLASS_NAMES}")
    print(f"训练集批次大小: {BATCH_SIZE}")
    print(f"每轮训练迭代数: {len(train_dataset) // BATCH_SIZE}")
    print("=" * 60)


def plot_class_distribution(train_dataset, test_dataset, save_path=None):
    """
    绘制训练集和测试集的类别分布直方图
    满足老师要求：利用图表展示数据分布
    """
    # 统计各类别数量
    train_labels = [label for _, label in train_dataset]
    test_labels = [label for _, label in test_dataset]

    train_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    test_counts = np.bincount(test_labels, minlength=NUM_CLASSES)

    x = np.arange(NUM_CLASSES)
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, train_counts, width, label='Training Set',
                   color='#4CAF50', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, test_counts, width, label='Test Set',
                   color='#2196F3', edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_title('CIFAR-10 Class Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱子上方显示数值
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"类别分布图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_pixel_distribution(dataset, save_path=None):
    """
    绘制像素值分布直方图
    展示RGB三个通道的像素值分布
    """
    # 收集所有图像的像素值（不归一化的原始数据）
    all_pixels = {0: [], 1: [], 2: []}
    channel_names = ['Red', 'Green', 'Blue']
    colors = ['#FF4444', '#44FF44', '#4444FF']

    # 只取前1000张图像避免内存溢出
    for i, (img, _) in enumerate(dataset):
        if i >= 1000:
            break
        for c in range(3):
            all_pixels[c].extend(img[c].flatten().numpy())

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for c, (ax, name, color) in enumerate(zip(axes, channel_names, colors)):
        ax.hist(all_pixels[c], bins=50, color=color, alpha=0.7,
                edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Pixel Value', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Channel', fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)

    plt.suptitle('CIFAR-10 Pixel Value Distribution (Normalized)',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"像素分布图已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_preprocessing_comparison(raw_dataset, save_path=None):
    """
    展示数据预处理前后的对比
    左侧：原始图像（反归一化后）
    右侧：经过数据增强后的图像
    """
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    # 反归一化参数
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    for i in range(8):
        img, label = raw_dataset[i]

        # 反归一化
        img_denorm = img * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_np = img_denorm.permute(1, 2, 0).numpy()

        # 显示处理后的图像
        row, col = i // 2, (i % 2) * 2
        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'Processed: {CLASS_NAMES[label]}', fontsize=9)
        axes[row, col].axis('off')

        # 显示原始像素值的统计信息
        axes[row, col + 1].hist(img_np.flatten(), bins=30, color='#2196F3',
                                 alpha=0.7, edgecolor='white')
        axes[row, col + 1].set_title(f'Pixel Distribution', fontsize=9)
        axes[row, col + 1].set_xlabel('Value', fontsize=8)
        axes[row, col + 1].set_ylabel('Freq', fontsize=8)
        axes[row, col + 1].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Data Preprocessing Visualization\n(Left: Processed Image, Right: Pixel Distribution)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预处理对比图已保存至: {save_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # 测试数据加载
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders()
    print_dataset_info(train_dataset, test_dataset)
    visualize_dataset_samples(train_dataset, save_path="../figures/cifar10_samples.png")
    plot_class_distribution(train_dataset, test_dataset, save_path="../figures/cifar10_class_distribution.png")
    plot_pixel_distribution(train_dataset, save_path="../figures/cifar10_pixel_distribution.png")
    plot_preprocessing_comparison(train_dataset, save_path="../figures/cifar10_preprocessing_comparison.png")
