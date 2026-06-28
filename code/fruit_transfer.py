# -*- coding: utf-8 -*-
"""
自构水果数据集 - ResNet-18 迁移学习模块

定位：
    本文件只负责“水果数据集加分实验”，不参与 CIFAR-10 主实验训练。

实验目标：
    使用已经构建好的 data/output_dataset/train 与 data/output_dataset/test，
    加载 ImageNet 预训练 ResNet-18，并通过“冻结大部分 backbone + 解冻 layer4 + FC”
    的方式完成五类水果分类迁移学习。

注意：
    - 本任务采用文件夹名称作为标签（folder-based labeling）。
    - 不再对水果数据 random_split，因为 dataset_builder.py 已经完成 80%/20% 均衡划分。
"""

import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FRUIT_DATA_DIR = BASE_DIR / "data" / "output_dataset"
DEFAULT_FIGURES_DIR = BASE_DIR / "figures" / "fruit_transfer"
DEFAULT_RESULTS_DIR = BASE_DIR / "results" / "fruit_transfer"
DEFAULT_MODEL_DIR = BASE_DIR / "models" / "fruit_transfer"

# ImageFolder 默认按文件夹名字母顺序映射类别：
# apple -> 0, banana -> 1, grape -> 2, orange -> 3, watermelon -> 4
EXPECTED_FRUIT_CLASSES = ["apple", "banana", "grape", "orange", "watermelon"]
FRUIT_CLASS_NAMES_ZH = {
    "apple": "苹果",
    "banana": "香蕉",
    "grape": "葡萄",
    "orange": "橙子",
    "watermelon": "西瓜",
}


def isolate_fruit_output_dir(path, default_dir):
    """
    将水果迁移学习结果隔离到独立子目录。

    目的：
        CIFAR-10 主实验结果保存在 figures/results/models 根目录；
        水果迁移学习加分实验保存在对应的 fruit_transfer 子目录，
        避免论文整理时把主实验和加分实验混在一起。
    """
    path = Path(path)
    if str(path) in {".", ""}:
        return Path(default_dir)
    if path.name == "fruit_transfer":
        return path
    if path.name in {"figures", "results", "models"}:
        return path / "fruit_transfer"
    return path


def get_fruit_loaders(data_dir=DEFAULT_FRUIT_DATA_DIR, batch_size=16, num_workers=0):
    """
    加载已经构建好的自构水果数据集。

    数据目录必须是标准 ImageFolder 结构：
        output_dataset/
            train/apple ...
            test/apple ...

    这里不再 random_split，避免破坏 dataset_builder.py 已完成的均衡 80%/20% 划分。
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            f"未找到水果数据集 train/test 目录：{data_dir}\n"
            "请先运行 data/dataset_builder.py 构建 output_dataset。"
        )

    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(root=str(train_dir), transform=transform_train)
    test_dataset = datasets.ImageFolder(root=str(test_dir), transform=transform_test)

    if train_dataset.classes != test_dataset.classes:
        raise ValueError(f"训练集和测试集类别不一致：{train_dataset.classes} vs {test_dataset.classes}")
    if train_dataset.classes != EXPECTED_FRUIT_CLASSES:
        print("警告：类别顺序与预期不完全一致，请以 ImageFolder 输出为准。")

    print("\n水果数据集信息:")
    print("  标注方式: 文件夹名称作为标签（folder-based labeling）")
    print(f"  数据目录: {data_dir}")
    print(f"  训练集目录: {train_dir}")
    print(f"  测试集目录: {test_dir}")
    print(f"  类别映射: {train_dataset.class_to_idx}")
    print(f"  训练集样本数: {len(train_dataset)}")
    print(f"  测试集样本数: {len(test_dataset)}")

    print("\n训练集类别统计:")
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    for idx, cls_name in enumerate(train_dataset.classes):
        train_count = int(np.sum(train_targets == idx))
        test_count = int(np.sum(test_targets == idx))
        print(f"  {cls_name:10s}: train={train_count:3d}, test={test_count:3d}")

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader, train_dataset, test_dataset, train_dataset.classes


def create_resnet18_transfer(num_classes=5, unfreeze_layer4=True):
    """
    创建用于迁移学习的 ResNet-18。

    策略：
        1. 加载 ImageNet 预训练权重；
        2. 冻结大部分 backbone，保留通用视觉特征；
        3. 解冻 layer4，允许高层语义特征适配水果数据；
        4. 替换 fc 层为五分类输出。
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    if unfreeze_layer4:
        for param in model.layer4.parameters():
            param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate_fruit_model(model, data_loader, criterion, device):
    """在测试集上评估模型，返回 loss、accuracy、真实标签和预测标签。"""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(data_loader), 1)
    acc = 100.0 * correct / max(total, 1)
    return avg_loss, acc, y_true, y_pred


def train_fruit_model(
    model,
    train_loader,
    test_loader,
    device,
    num_epochs=15,
    lr=3e-4,
    weight_decay=1e-4,
    model_save_path=None,
):
    """
    训练水果数据集迁移学习模型。

    参数选择：
        - epoch=15：小数据集上比 10 轮更稳，但不会明显过长；
        - AdamW + weight_decay：比普通 Adam 更利于抑制小样本过拟合；
        - CosineAnnealingLR：与主实验风格一致，学习率平滑衰减；
        - 解冻 layer4 + fc：比只训练 fc 更有适配能力。
    """
    print(f"\n{'=' * 70}")
    print("水果数据集迁移学习训练（ImageNet ResNet-18 -> 5类水果）")
    print(f"{'=' * 70}")
    print(f"设备: {device}")
    print(f"训练轮数: {num_epochs}")
    print(f"学习率: {lr}")
    print(f"权重衰减: {weight_decay}")
    print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
    print("微调策略: 冻结 backbone，解冻 layer4 + fc")
    print(f"{'=' * 70}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    model.to(device)
    history = {
        "model_name": "Fruit_ResNet18_Transfer",
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
        "epochs": [],
        "best_acc": 0.0,
        "best_epoch": 0,
    }

    best_acc = 0.0
    best_epoch = 0
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_total = 0
        train_correct = 0

        pbar = tqdm(train_loader, desc=f"Fruit Epoch {epoch + 1}/{num_epochs}", leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * train_correct / max(train_total, 1):.1f}%",
            })

        train_loss_avg = train_loss / max(len(train_loader), 1)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        val_loss, val_acc, _, _ = evaluate_fruit_model(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(train_loss_avg)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)
        history["epochs"].append(epoch + 1)

        print(
            f"Epoch [{epoch + 1:2d}/{num_epochs}] "
            f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1f}% | LR: {current_lr:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            if model_save_path is not None:
                os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                torch.save(
                    {
                        "epoch": best_epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_acc,
                    },
                    model_save_path,
                )
                print(f"  -> 最佳水果模型已保存 (Acc: {best_acc:.2f}%)")

    history["best_acc"] = best_acc
    history["best_epoch"] = best_epoch
    history["total_time"] = time.time() - start_time

    print(f"\n训练完成！最佳验证准确率: {best_acc:.2f}% (epoch {best_epoch})")
    print(f"总耗时: {history['total_time'] / 60:.2f} 分钟")
    return history


def plot_fruit_training_curves(history, save_path=None):
    """绘制水果迁移学习训练曲线。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    epochs = history["epochs"]
    axes[0].plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], marker="s", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Fruit Transfer Learning Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], marker="o", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], marker="s", label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Fruit Transfer Learning Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.suptitle("ResNet-18 Transfer Learning on Self-built Fruit Dataset", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"水果迁移学习训练曲线已保存至: {save_path}")
    plt.close(fig)


def visualize_fruit_predictions(model, test_loader, class_names, device, save_path=None, max_images=20):
    """
    可视化水果测试集预测结果。

    图中绿色标题表示预测正确，红色标题表示预测错误，可直接用于论文展示。
    """
    model.eval()
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)

            images_list.extend(images.cpu())
            labels_list.extend(labels.numpy())
            preds_list.extend(preds.cpu().numpy())

            if len(images_list) >= max_images:
                break

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    cols = 5
    rows = int(np.ceil(min(max_images, len(images_list)) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(3.0, rows * 3.0)))
    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis("off")

    for i in range(min(max_images, len(images_list))):
        img = images_list[i] * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        true_label = class_names[labels_list[i]]
        pred_label = class_names[preds_list[i]]
        is_correct = labels_list[i] == preds_list[i]
        color = "green" if is_correct else "red"

        axes[i].imshow(img)
        axes[i].set_title(f"T: {true_label}\nP: {pred_label}", color=color, fontsize=9)
        axes[i].axis("off")

    plt.suptitle("Fruit Dataset Predictions (Green=Correct, Red=Wrong)", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"水果预测结果图已保存至: {save_path}")
    plt.close(fig)


def save_fruit_reports(history, model, test_loader, class_names, device, results_dir):
    """保存水果实验训练历史、分类报告和混淆矩阵数据。"""
    os.makedirs(results_dir, exist_ok=True)
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc, y_true, y_pred = evaluate_fruit_model(model, test_loader, criterion, device)

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred).tolist()

    history_path = os.path.join(results_dir, "fruit_transfer_history.json")
    report_path = os.path.join(results_dir, "fruit_classification_report.json")
    cm_path = os.path.join(results_dir, "fruit_confusion_matrix.json")

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "class_names": list(class_names),
                "class_name_zh": {name: FRUIT_CLASS_NAMES_ZH.get(name, name) for name in class_names},
                "test_loss": val_loss,
                "test_acc": val_acc,
                "classification_report": report_dict,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    with open(cm_path, "w", encoding="utf-8") as f:
        json.dump({"class_names": list(class_names), "confusion_matrix": cm}, f, ensure_ascii=False, indent=2)

    print(f"水果训练历史已保存至: {history_path}")
    print(f"水果分类报告已保存至: {report_path}")
    print(f"水果混淆矩阵数据已保存至: {cm_path}")

    return val_acc


def run_fruit_experiment(
    data_dir=DEFAULT_FRUIT_DATA_DIR,
    figures_dir=DEFAULT_FIGURES_DIR,
    results_dir=DEFAULT_RESULTS_DIR,
    model_dir=DEFAULT_MODEL_DIR,
    device=None,
    num_epochs=15,
    batch_size=16,
    lr=3e-4,
    num_workers=0,
):
    """
    运行完整水果迁移学习实验：
        加载标准 output_dataset -> 创建预训练 ResNet-18 -> 微调 -> 保存图表/模型/报告。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figures_dir = isolate_fruit_output_dir(figures_dir, DEFAULT_FIGURES_DIR)
    results_dir = isolate_fruit_output_dir(results_dir, DEFAULT_RESULTS_DIR)
    model_dir = isolate_fruit_output_dir(model_dir, DEFAULT_MODEL_DIR)

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("【自构水果数据集 - ResNet-18 迁移学习实验】")
    print("=" * 70)
    print("输出目录已与 CIFAR-10 主实验隔离：")
    print(f"  水果图表目录: {figures_dir}")
    print(f"  水果结果目录: {results_dir}")
    print(f"  水果模型目录: {model_dir}")

    train_loader, test_loader, train_dataset, test_dataset, class_names = get_fruit_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print("\n-> 创建 ResNet-18（ImageNet 预训练，解冻 layer4 + fc）...")
    model = create_resnet18_transfer(num_classes=len(class_names), unfreeze_layer4=True)

    model_path = os.path.join(model_dir, "fruit_resnet18_best.pth")
    history = train_fruit_model(
        model,
        train_loader,
        test_loader,
        device,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=1e-4,
        model_save_path=model_path,
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    print("\n-> 生成水果迁移学习训练曲线...")
    plot_fruit_training_curves(
        history,
        save_path=os.path.join(figures_dir, "fruit_transfer_training_curves.png"),
    )

    print("\n-> 生成水果测试集预测结果可视化...")
    visualize_fruit_predictions(
        model,
        test_loader,
        class_names,
        device,
        save_path=os.path.join(figures_dir, "fruit_predictions.png"),
    )

    final_acc = save_fruit_reports(history, model, test_loader, class_names, device, results_dir)

    print("\n" + "=" * 70)
    print("水果数据集迁移学习实验完成！")
    print(f"最佳验证准确率: {history['best_acc']:.2f}%")
    print(f"最终测试准确率: {final_acc:.2f}%")
    print(f"最佳模型保存至: {model_path}")
    print("=" * 70)

    return history, history["best_acc"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_fruit_experiment(device=device)
