# CIFAR-10 图像分类：ClassicCNN 与 ResNet-18 对比研究

## 项目简介

本项目为《人工智能》课程论文实验部分，基于 PyTorch 框架实现，对 CIFAR-10 数据集进行图像分类任务，对比分析了\*\*经典卷积神经网络（ClassicCNN）\*\*与 **ResNet-18** 两种模型的性能差异。

## 实验内容

### 数据集

* **CIFAR-10**：包含 60,000 张 32x32 彩色图像，分为 10 个类别
* 训练集：50,000 张，测试集：10,000 张
* 10 个类别：airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 数据预处理

* **训练集**：随机裁剪（padding=4）→ 随机水平翻转 → ToTensor → 归一化（均值=\[0.4914, 0.4822, 0.4465]，标准差=\[0.2470, 0.2435, 0.2616]）
* **测试集**：ToTensor → 归一化
* 目的：数据增强提高模型泛化能力，归一化加速训练收敛

### 模型架构

#### 1\. ClassicCNN（经典卷积神经网络）

* 3 层卷积块（Conv + BatchNorm + ReLU + MaxPool）
* 2 层全连接层 + Dropout
* 参数量：约 120 万

#### 2\. ResNet-18（残差网络）

* 4 组残差块（每组 2 个 BasicBlock）
* 跳跃连接（Skip Connection）解决梯度消失问题
* 全局平均池化 + 全连接层
* 参数量：约 1,100 万

### 评估指标

* 准确率（Accuracy）
* 精确率（Precision）
* 召回率（Recall）
* F1 分数（F1-Score）
* 敏感性（Sensitivity）
* 特异性（Specificity）
* ROC 曲线与 AUC 值
* 混淆矩阵

## 环境要求

* Python >= 3.8
* PyTorch >= 1.10（推荐 GPU 版本）
* torchvision >= 0.11
* scikit-learn >= 1.0
* matplotlib >= 3.4
* numpy >= 1.21
* tqdm >= 4.62

## 安装依赖

```bash
pip install -r requirements.txt
```

### GPU 环境（推荐）

如需使用 GPU 加速，请安装 CUDA 版本的 PyTorch：

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

验证 GPU 是否可用：

```bash
python -c "import torch; print(torch.cuda.is\_available())"
```

## 项目结构

```
ai\_course\_project/
├── src/                       # 源代码目录
│   ├── config.py              # 配置文件（超参数、路径、设备）
│   ├── models.py              # 模型定义（ClassicCNN、ResNet-18）
│   ├── data\_utils.py          # 数据加载、预处理、可视化
│   ├── train.py               # 训练模块（训练、验证、保存）
│   ├── evaluate.py            # 评估模块（指标计算、可视化）
│   └── main.py                # 主程序入口
├── data/                      # 数据集存放目录（自动下载）
├── models/                    # 保存的训练模型
├── figures/                   # 生成的可视化图表
├── results/                   # 训练历史记录（JSON）
├── requirements.txt           # 项目依赖
└── README.md                  # 项目说明
```

## 使用方法

### 一键运行完整实验

```bash
cd src
python main.py
```

程序将自动完成以下步骤：

1. 下载 CIFAR-10 数据集（首次运行，约 170MB）
2. 生成数据分布可视化（样本图、直方图、像素分布图、预处理对比图）
3. 训练 ClassicCNN 模型
4. 训练 ResNet-18 模型
5. 生成所有评估图表和对比分析

### 单独运行各模块

查看模型结构：

```bash
cd src
python -c "from models import ClassicCNN, print\_model\_summary; from config import DEVICE; import torch; m = ClassicCNN(10).to(DEVICE); print\_model\_summary(m, 'ClassicCNN')"
```

## 实验结果

运行完成后，`figures/` 目录将包含以下图表（共约 18 张）：

|图表|说明|论文章节|
|-|-|-|
|cifar10\_samples.png|每类样本展示|数据介绍|
|cifar10\_class\_distribution.png|类别分布直方图|数据介绍|
|cifar10\_pixel\_distribution.png|像素值分布图|数据介绍|
|cifar10\_preprocessing\_comparison.png|预处理对比图|数据介绍|
|training\_curves\_comparison.png|训练曲线对比|实验结果|
|\*\_confusion\_matrix.png|混淆矩阵|定量评估|
|\*\_per\_class\_metrics.png|各类别精确率/召回率/F1|定量评估|
|\*\_sens\_spec.png|敏感性/特异性|定量评估|
|\*\_roc\_curves.png|ROC曲线与AUC|定量评估|
|\*\_predictions.png|预测结果可视化|定性评估|
|\*\_error\_analysis.png|错误案例分析|定性评估|

## 论文写作参考

### 图片来源标注格式

所有代码生成的图表，图片来源标注为：

```
图X 标题（图片来源：本实验代码生成）
```

自行绘制的网络结构图，图片来源标注为：

```
图X 标题（图片来源：使用PowerPoint/Visio自行绘制）
```

### 来源说明示例

```
本文选题为"基于卷积神经网络的CIFAR-10图像分类任务"。实验代码参考PyTorch官方
文档及ResNet原始论文（He et al., 2016）实现，模型结构在理解原理的基础上自行
搭建。数据预处理流程参考CIFAR-10标准处理方法。论文写作过程中使用大语言模型
辅助进行资料检索、概念解释和文字润色，实验过程及结果分析由本人独立完成。
```

## 参考文献

1. He, K., Zhang, X., Ren, S., \& Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR.
2. Krizhevsky, A., \& Hinton, G. (2009). Learning Multiple Layers of Features from Tiny Images. Technical Report, University of Toronto.
3. PyTorch官方文档：https://pytorch.org/docs/stable/index.html
4. torchvision文档：https://pytorch.org/vision/stable/index.html

