"""
模型定义模块
包含：经典CNN 和 ResNet-18
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. 经典CNN模型（3层卷积 + 2层全连接）
# ==========================================
class ClassicCNN(nn.Module):
    """
    经典卷积神经网络结构
    结构：Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC
    """
    def __init__(self, num_classes=10):
        super(ClassicCNN, self).__init__()
        # 第1个卷积块: 3x32x32 -> 32x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)

        # 第2个卷积块: 32x16x16 -> 64x16x16 -> 64x8x8
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # 第3个卷积块: 64x8x8 -> 128x8x8 -> 128x4x4
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # 全连接层: 128*4*4 = 2048 -> 512 -> num_classes
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 卷积块1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        # 卷积块2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # 卷积块3
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        # 展平
        x = x.view(x.size(0), -1)
        # 全连接
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ==========================================
# 2. ResNet-18 模型
# ==========================================
class BasicBlock(nn.Module):
    """
    ResNet基础残差块
    包含两个3x3卷积层和跳跃连接（shortcut connection）
    """
    expansion = 1  # 输出通道数相对于输入通道数的倍数

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层，可能进行下采样
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # 下采样层（用于调整输入维度以匹配输出）
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # 保存输入用于跳跃连接

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果需要进行下采样，调整identity的维度
        if self.downsample is not None:
            identity = self.downsample(x)

        # 跳跃连接：输出 = 卷积结果 + 输入
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet网络主体结构
    """
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        # 初始卷积层: 3x32x32 -> 64x32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 注意：CIFAR-10图像较小(32x32)，不使用7x7卷积和初始池化

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])   # 64x32x32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 128x16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 256x8x8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 512x4x4

        # 全局平均池化 + 全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        self._initialize_weights()

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        """构建一个残差层，包含多个残差块"""
        downsample = None
        # 当stride不为1或通道数变化时，需要下采样
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet18(num_classes=10):
    """创建ResNet-18模型"""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def print_model_summary(model, model_name="Model"):
    """
    打印模型结构摘要
    用于论文中描述模型结构
    """
    print("\n" + "=" * 60)
    print(f"{model_name} 结构摘要")
    print("=" * 60)

    total_params = 0
    trainable_params = 0

    print(f"{'Layer':<30s} {'Output Shape':<20s} {'Params':<15s}")
    print("-" * 65)

    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            trainable_params += params
        total_params += param.numel()

    # 按模块打印
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        print(f"{name:<30s} {str(tuple(module.parameters()).__len__()) + ' params':<20s} {module_params:<15,}")

    print("-" * 65)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    print("=" * 60)

    return total_params, trainable_params


if __name__ == "__main__":
    # 测试模型输出尺寸
    from config import DEVICE

    # 测试ClassicCNN
    model_cnn = ClassicCNN(num_classes=10).to(DEVICE)
    test_input = torch.randn(2, 3, 32, 32).to(DEVICE)
    output = model_cnn(test_input)
    print(f"ClassicCNN 输出尺寸: {output.shape}")
    print_model_summary(model_cnn, "ClassicCNN")

    # 测试ResNet-18
    model_resnet = resnet18(num_classes=10).to(DEVICE)
    output = model_resnet(test_input)
    print(f"\nResNet-18 输出尺寸: {output.shape}")
    print_model_summary(model_resnet, "ResNet-18")
