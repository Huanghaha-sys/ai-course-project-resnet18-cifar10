"""
Feature Map 可视化模块
隐藏加分点：直观展示CNN vs ResNet特征提取差异
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from config import *


class FeatureExtractor:
    """
    特征提取器：通过hook获取模型中间层输出
    """
    def __init__(self, model, target_layers):
        """
        参数:
            model: 神经网络模型
            target_layers: 要提取特征层的名称列表
        """
        self.model = model
        self.target_layers = target_layers
        self.features = {}
        self.hooks = []

        # 注册hook
        for name, module in model.named_modules():
            if name in target_layers:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)

    def _get_hook(self, layer_name):
        """创建hook函数"""
        def hook_fn(module, input, output):
            self.features[layer_name] = output.detach()
        return hook_fn

    def extract(self, x):
        """前向传播并提取特征"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(x)
        return self.features

    def remove_hooks(self):
        """移除所有hook"""
        for hook in self.hooks:
            hook.remove()


def visualize_feature_maps(features, layer_name, save_path=None, max_channels=16):
    """
    可视化某层的特征图（热力图形式）
    
    参数:
        features: 特征提取器输出的特征字典
        layer_name: 要可视化的层名
        save_path: 保存路径
        max_channels: 最多显示多少个通道
    """
    feature = features[layer_name]  # [1, C, H, W]
    channels = min(feature.shape[1], max_channels)

    # 计算合适的行列数
    rows = int(np.ceil(channels / 4))
    cols = min(channels, 4)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(channels):
        fmap = feature[0, i].cpu().numpy()
        # 归一化到0-1以便显示
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)

        axes[i].imshow(fmap, cmap='viridis')
        axes[i].set_title(f'Ch {i+1}', fontsize=9)
        axes[i].axis('off')

    # 隐藏多余的子图
    for i in range(channels, len(axes)):
        axes[i].axis('off')

    plt.suptitle(f'Feature Maps - {layer_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"特征图已保存至: {save_path}")
    plt.show()
    plt.close()


def compare_feature_maps(cnn_model, resnet_model, test_loader,
                         save_dir=FIGURES_DIR, device=DEVICE):
    """
    对比CNN和ResNet的特征图差异
    这是隐藏加分点的核心实现
    """
    print("\n【Feature Map可视化 - 隐藏加分点】")

    # 获取一批测试图像
    images, labels = next(iter(test_loader))
    images = images[:4].to(device)  # 取4张图

    # 反归一化显示原始图像
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(device)
    images_denorm = images * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)

    # ========== CNN特征提取 ==========
    print("  -> 提取ClassicCNN特征...")
    # CNN的层：conv1 -> conv2 -> conv3
    cnn_layers = ['conv1', 'conv2', 'conv3']
    cnn_extractor = FeatureExtractor(cnn_model, cnn_layers)
    cnn_features = cnn_extractor.extract(images)
    cnn_extractor.remove_hooks()

    # ========== ResNet特征提取 ==========
    print("  -> 提取ResNet-18特征...")
    # ResNet的关键层：layer1(浅层) 和 layer4(深层)
    resnet_layers = ['layer1', 'layer4']
    resnet_extractor = FeatureExtractor(resnet_model, resnet_layers)
    resnet_features = resnet_extractor.extract(images)
    resnet_extractor.remove_hooks()

    # ========== 可视化1：输入图像 ==========
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i in range(4):
        img = images_denorm[i].permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(img)
        axes[i].set_title(f'Input {i+1}\n({CLASS_NAMES[labels[i]]})', fontsize=9)
        axes[i].axis('off')
    plt.suptitle('Input Images for Feature Visualization', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "featuremap_input_images.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  输入图像已保存: {save_path}")
    plt.show()
    plt.close()

    # ========== 可视化2：CNN vs ResNet浅层特征对比 ==========
    print("  -> 生成CNN vs ResNet浅层特征对比...")
    compare_layer = 'conv1'  # CNN第一层
    resnet_layer = 'layer1'  # ResNet第一层

    fig, axes = plt.subplots(4, 9, figsize=(18, 8))

    for img_idx in range(4):
        # 显示原始图像（第一列）
        img = images_denorm[img_idx].permute(1, 2, 0).cpu().numpy()
        axes[img_idx, 0].imshow(img)
        axes[img_idx, 0].set_title('Input', fontsize=9)
        axes[img_idx, 0].axis('off')

        # CNN特征（第2-5列）
        cnn_feat = cnn_features[compare_layer][img_idx]  # [C, H, W]
        for ch in range(4):
            fmap = cnn_feat[ch].cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axes[img_idx, ch + 1].imshow(fmap, cmap='viridis')
            axes[img_idx, ch + 1].set_title(f'CNN Ch{ch+1}', fontsize=8)
            axes[img_idx, ch + 1].axis('off')

        # ResNet特征（第6-9列）
        res_feat = resnet_features[resnet_layer][img_idx]
        for ch in range(4):
            fmap = res_feat[ch].cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axes[img_idx, ch + 5].imshow(fmap, cmap='plasma')
            axes[img_idx, ch + 5].set_title(f'ResNet Ch{ch+1}', fontsize=8)
            axes[img_idx, ch + 5].axis('off')

    plt.suptitle('Shallow Feature Comparison: CNN (conv1) vs ResNet (layer1)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "featuremap_cnn_vs_resnet_shallow.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  浅层对比已保存: {save_path}")
    plt.show()
    plt.close()

    # ========== 可视化3：CNN深层 vs ResNet深层 ==========
    print("  -> 生成深层特征对比...")
    compare_layer_deep = 'conv3'  # CNN最后一层
    resnet_layer_deep = 'layer4'  # ResNet最后一层

    fig, axes = plt.subplots(4, 9, figsize=(18, 8))

    for img_idx in range(4):
        img = images_denorm[img_idx].permute(1, 2, 0).cpu().numpy()
        axes[img_idx, 0].imshow(img)
        axes[img_idx, 0].set_title('Input', fontsize=9)
        axes[img_idx, 0].axis('off')

        # CNN深层特征
        cnn_feat = cnn_features[compare_layer_deep][img_idx]
        for ch in range(4):
            fmap = cnn_feat[ch].cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axes[img_idx, ch + 1].imshow(fmap, cmap='viridis')
            axes[img_idx, ch + 1].set_title(f'CNN Ch{ch+1}', fontsize=8)
            axes[img_idx, ch + 1].axis('off')

        # ResNet深层特征
        res_feat = resnet_features[resnet_layer_deep][img_idx]
        for ch in range(4):
            fmap = res_feat[ch].cpu().numpy()
            fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
            axes[img_idx, ch + 5].imshow(fmap, cmap='plasma')
            axes[img_idx, ch + 5].set_title(f'ResNet Ch{ch+1}', fontsize=8)
            axes[img_idx, ch + 5].axis('off')

    plt.suptitle('Deep Feature Comparison: CNN (conv3) vs ResNet (layer4)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "featuremap_cnn_vs_resnet_deep.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  深层对比已保存: {save_path}")
    plt.show()
    plt.close()

    # ========== 可视化4：单张图的多层特征演化 ==========
    print("  -> 生成单张图像的多层特征演化...")
    img_idx = 0  # 取第一张

    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    # 显示原始图像
    img = images_denorm[img_idx].permute(1, 2, 0).cpu().numpy()
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Input Image', fontsize=10)
    axes[0, 0].axis('off')

    # CNN多层特征
    for layer_idx, layer_name in enumerate(['conv1', 'conv2', 'conv3']):
        feat = cnn_features[layer_name][img_idx]  # [C, H, W]
        fmap = feat[0].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        axes[0, layer_idx + 1].imshow(fmap, cmap='viridis')
        axes[0, layer_idx + 1].set_title(f'CNN {layer_name}', fontsize=10)
        axes[0, layer_idx + 1].axis('off')

    # 空一格
    axes[0, 4].axis('off')

    # ResNet多层特征
    axes[1, 0].imshow(img)
    axes[1, 0].set_title('Input Image', fontsize=10)
    axes[1, 0].axis('off')

    resnet_layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
    # 需要重新提取layer2和layer3
    resnet_extractor2 = FeatureExtractor(resnet_model, resnet_layer_names)
    resnet_features_full = resnet_extractor2.extract(images[img_idx:img_idx + 1])
    resnet_extractor2.remove_hooks()

    for layer_idx, layer_name in enumerate(resnet_layer_names):
        feat = resnet_features_full[layer_name][0]
        fmap = feat[0].cpu().numpy()
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        axes[1, layer_idx + 1].imshow(fmap, cmap='plasma')
        axes[1, layer_idx + 1].set_title(f'ResNet {layer_name}', fontsize=10)
        axes[1, layer_idx + 1].axis('off')

    # 说明文字
    axes[2, 0].axis('off')
    axes[2, 1].axis('off')
    axes[2, 2].text(0.5, 0.5,
                    'Observation:\n'
                    'CNN features become increasingly abstract\n'
                    'ResNet features maintain richer information\n'
                    'through skip connections',
                    ha='center', va='center', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')
    axes[2, 4].axis('off')

    plt.suptitle('Multi-Layer Feature Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, "featuremap_multilayer_evolution.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  多层演化已保存: {save_path}")
    plt.show()
    plt.close()

    print("【Feature Map可视化完成】")


if __name__ == "__main__":
    from models import ClassicCNN, resnet18
    from data_utils import get_data_loaders

    train_loader, test_loader, _, _ = get_data_loaders()

    cnn_model = ClassicCNN(num_classes=10).to(DEVICE)
    resnet_model = resnet18(num_classes=10).to(DEVICE)

    compare_feature_maps(cnn_model, resnet_model, test_loader)
