"""
配置文件：包含所有超参数和路径设置
"""
import os
import torch

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据集配置
DATA_ROOT = "./data"
NUM_CLASSES = 10  # CIFAR-10有10个类别
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 训练配置
BATCH_SIZE = 64       # CPU训练用较小的batch
NUM_EPOCHS = 30       # 训练轮数（ResNet-18建议30轮以上）
LEARNING_RATE = 0.001  # 学习率
NUM_WORKERS = 2       # 数据加载线程数

# 模型配置
RESNET_DEPTH = 18     # ResNet深度: 18

# 路径配置
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"
MODEL_SAVE_DIR = "./models"

# 创建必要的目录
for d in [RESULTS_DIR, FIGURES_DIR, MODEL_SAVE_DIR]:
    os.makedirs(d, exist_ok=True)

print(f"设备: {DEVICE}")
print(f"批次大小: {BATCH_SIZE}")
print(f"训练轮数: {NUM_EPOCHS}")
