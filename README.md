## **基于 ResNet-18 的 CIFAR-10 图像分类算法复现与迁移学习应用验证**

###### &#x20; 

《人工智能》课程论文配套实验工程
  ResNet-18 复现 ｜ ClassicCNN 对照实验 ｜ 自构水果数据集 ｜ 迁移学习验证
---

###### 

#### 项目简介

本项目为《人工智能》课程论文配套实验工程，围绕“基于 ResNet-18 的图像分类算法复现与性能分析”展开，使用 PyTorch 完成了以下两部分实验：

1. CIFAR-10 主体实验：以 ResNet-18 为主模型，以自行设计的 ClassicCNN 为对照模型，完成图像分类训练、测试与可视化分析。
2. 自构水果数据集加分实验：基于 Kaggle 水果蔬菜图像数据构建 5 类水果分类数据集，并使用 ImageNet 预训练 ResNet-18 完成迁移学习验证。

本项目不仅包含模型训练代码，还包含数据预处理、数据集构建、评价指标统计、特征图可视化、错误案例分析、迁移学习实验与论文配套图表生成代码，可作为课程论文实验部分的完整工程实现。



#### 实验内容

###### 1\. CIFAR-10 主体实验

* 数据集：CIFAR-10，共 60,000 张 32×32 彩色图像，10 个类别。
* 主模型：ResNet-18。
* 对照模型：ClassicCNN（三层卷积神经网络 baseline）。
* 数据增强：RandomCrop、RandomHorizontalFlip、Normalize。
* 评价指标：Accuracy、Precision、Recall、F1、ROC、AUC、混淆矩阵。
* 定性分析：预测结果展示、错误案例分析、CNN 与 ResNet 特征图对比。



###### 2\. 自构水果数据集迁移学习实验

* 原始数据来源：Kaggle Fruit and Vegetable Image Recognition dataset。
* 目标类别：`apple`、`banana`、`orange`、`grape`、`watermelon`。
* 数据工程流程：类别筛选、损坏图片检查、异常图像过滤、RGB 转换、统一 resize、hash 去重、类别均衡划分、可视化与统计报告输出。
* 标注方式：采用文件夹名称作为类别标签，即 folder-based labeling。
* 迁移学习模型：加载 ImageNet 预训练权重的 ResNet-18，冻结大部分 backbone，仅解冻 `layer4 + fc` 进行微调。



#### 关键结果

###### 1\. CIFAR-10 结果

|模型|最佳测试准确率|
|-|-|
|ClassicCNN|84.87%|
|ResNet-18|93.04%|

ResNet-18 相比 ClassicCNN 的准确率提升为 8.17 个百分点，说明残差连接在更深层网络训练中能够显著改善特征表达能力与分类性能。



###### 2\. 自构水果数据集结果

|指标|数值|
|-|-|
|测试准确率|98.46%|
|Macro Precision|0.9857|
|Macro Recall|0.9846|
|Macro F1|0.9846|

该结果表明，基于 ImageNet 预训练权重的 ResNet-18 能够在小样本水果分类任务上取得较高精度，验证了迁移学习在课程实验场景中的可行性。



#### 自构水果数据集统计

###### 水果数据集由 `src/data/dataset\_builder.py` 构建完成，最终统计如下：

* 原始扫描图片总数：3825
* 非目标类别图片：3333
* 五类目标图片清洗前总数：492
* hash 去重删除：108
* 清洗后保留：325
* 训练集规模：260
* 测试集规模：65
* 划分方式：每类训练集 52 张，测试集 13 张，保持类别均衡
* 统一尺寸：224×224
* 最小尺寸阈值：50×50
* 

###### 各类别清洗前后数量如下：

|类别|清洗前|清洗后|
|-|-|-|
|apple|88|71|
|banana|93|73|
|orange|88|65|
|grape|119|95|
|watermelon|104|80|



#### 项目结构

本地项目采用 `src/` 作为主代码目录；若 GitHub 中为了展示进行了目录整理，请以仓库中的实际目录名称为准。

```text
ai\_course\_project/
├── src/
│   ├── config.py
│   ├── data\_utils.py
│   ├── evaluate.py
│   ├── feature\_map.py
│   ├── fruit\_transfer.py
│   ├── main.py
│   ├── models.py
│   ├── train.py
│   ├── data/
│   │   ├── dataset\_builder.py
│   │   ├── figures/
│   │   ├── output\_dataset/
│   │   ├── fruit\_dataset\_counts.csv
│   │   └── fruit\_dataset\_report.json
│   ├── figures/
│   │   └── fruit\_transfer/
│   ├── results/
│   │   └── fruit\_transfer/
│   └── models/
├── requirements.txt
└── README.md
```

#### 环境要求

* Python >= 3.8
* torch >= 1.10.0
* torchvision >= 0.11.0
* numpy >= 1.21.0
* matplotlib >= 3.4.0
* scikit-learn >= 1.0.0
* tqdm >= 4.62.0



#### 安装依赖：

```bash
pip install -r requirements.txt
```

如需使用 GPU，建议根据本机 CUDA 版本安装对应的 PyTorch。

<h2>使用方法</h2>

<h3>1. 运行 CIFAR-10 主体实验</h3>

```bash
cd src
python main.py
```

该脚本将完成：

1. 加载 CIFAR-10 数据集
2. 生成数据可视化图表
3. 训练 ClassicCNN
4. 训练 ResNet-18
5. 输出评估结果与特征图可视化
6. 在已存在水果数据集时自动运行迁移学习实验



#### 2\. 单独构建水果分类数据集

将 Kaggle 原始水果数据放入 `src/data/archive/` 后，运行：

```bash
cd src/data
python dataset\_builder.py --check\_loader
```

该脚本将完成：

1. 筛选 5 类目标水果
2. 删除损坏图片与异常图片
3. 转换为 RGB 三通道并统一 resize 为 224×224
4. 进行 hash 去重
5. 按 80%/20% 自动划分训练集与测试集
6. 生成 `figures/` 可视化图和 `JSON/CSV` 统计报告



#### 3\. 单独运行水果迁移学习实验

在已生成 `src/data/output\_dataset/` 后，运行：

```bash
cd src
python fruit\_transfer.py
```

输出将保存到：

* 图表目录：`src/figures/fruit\_transfer/`
* 结果目录：`src/results/fruit\_transfer/`
* 模型目录：`src/models/fruit\_transfer/`



#### 主要输出文件

#### CIFAR-10 图表

* `cifar10\_samples.png`
* `cifar10\_class\_distribution.png`
* `cifar10\_pixel\_distribution.png`
* `cifar10\_preprocessing\_comparison.png`
* `training\_curves\_comparison.png`
* `ClassicCNN\_confusion\_matrix.png`
* `ResNet18\_confusion\_matrix.png`
* `ResNet18\_per\_class\_metrics.png`
* `ResNet18\_roc\_curves.png`
* `ResNet18\_predictions.png`
* `ResNet18\_error\_analysis.png`
* `featuremap\_input\_images.png`
* `featuremap\_cnn\_vs\_resnet\_shallow.png`
* `featuremap\_cnn\_vs\_resnet\_deep.png`
* `featuremap\_multilayer\_evolution.png`

#### 水果数据集构建图表

* `src/data/figures/fruit\_samples\_grid.png`
* `src/data/figures/fruit\_class\_distribution.png`
* `src/data/figures/fruit\_cleaning\_before\_after.png`

#### 水果迁移学习图表与结果

* `src/figures/fruit\_transfer/fruit\_transfer\_training\_curves.png`
* `src/figures/fruit\_transfer/fruit\_predictions.png`
* `src/results/fruit\_transfer/fruit\_transfer\_history.json`
* `src/results/fruit\_transfer/fruit\_classification\_report.json`
* `src/results/fruit\_transfer/fruit\_confusion\_matrix.json`



#### 代码来源与说明

本文选题为“基于 ResNet-18 的 CIFAR-10 图像分类算法复现与迁移学习应用验证”。论文算法原理主要参考 He 等人提出的 ResNet 原始论文《Deep Residual Learning for Image Recognition》。实验代码由本人在参考 PyTorch 官方 CIFAR-10 分类教程、PyTorch 官方迁移学习教程、torchvision ResNet-18 模型文档以及 GitHub 开源项目 `kuangliu/pytorch-cifar` 的基础上完成，并结合课程论文要求自行实现了 ClassicCNN 对照模型、CIFAR-10 版 ResNet-18 训练流程、结果评估、特征图可视化、自构水果数据集构建和迁移学习模块。

论文写作过程中使用大语言模型辅助进行文字润色、结构整理、公式表述与代码注释优化，实验运行、结果生成、图片整理与论文最终修改由本人完成。

###### 参考资料

1. He, K., Zhang, X., Ren, S., \& Sun, J. Deep Residual Learning for Image Recognition. CVPR, 2016.
2. Krizhevsky, A., Hinton, G. Learning Multiple Layers of Features from Tiny Images. University of Toronto, 2009.
3. PyTorch Tutorials: Training a Classifier  
https://docs.pytorch.org/tutorials/beginner/blitz/cifar10\_tutorial.html
4. PyTorch Tutorials: Transfer Learning for Computer Vision Tutorial  
https://docs.pytorch.org/tutorials/beginner/transfer\_learning\_tutorial.html
5. torchvision ResNet-18 文档  
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
6. kuangliu/pytorch-cifar  
https://github.com/kuangliu/pytorch-cifar
7. Kaggle Fruit and Vegetable Image Recognition Dataset  
https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition



项目源码仓库地址：  
https://github.com/Huanghaha-sys/ai-course-project-resnet18-cifar10

