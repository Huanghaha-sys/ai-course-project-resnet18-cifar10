"""
评估与可视化模块
包含：训练曲线绘制、混淆矩阵、ROC曲线、预测结果可视化、错误案例分析
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report,
                             roc_curve, auc, precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize
import json
import os
from config import *


def plot_training_curves(histories, save_path=None):
    """
    绘制训练曲线对比图（支持多个模型对比）
    
    参数:
        histories: 训练历史字典列表
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ['#2196F3', '#FF5722']
    markers = ['o', 's']

    for idx, history in enumerate(histories):
        name = history['model_name']
        epochs = history['epochs']
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        # 损失曲线
        axes[0].plot(epochs, history['train_loss'], color=color,
                     marker=marker, markersize=4, linestyle='-',
                     label=f'{name} (Train)', alpha=0.8)
        axes[0].plot(epochs, history['val_loss'], color=color,
                     marker=marker, markersize=4, linestyle='--',
                     label=f'{name} (Val)', alpha=0.6)

        # 准确率曲线
        axes[1].plot(epochs, history['train_acc'], color=color,
                     marker=marker, markersize=4, linestyle='-',
                     label=f'{name} (Train)', alpha=0.8)
        axes[1].plot(epochs, history['val_acc'], color=color,
                     marker=marker, markersize=4, linestyle='--',
                     label=f'{name} (Val)', alpha=0.6)

    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Model Training Comparison', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"训练曲线已保存至: {save_path}")
    plt.show()
    plt.close()


def evaluate_model(model, test_loader, device=DEVICE):
    """
    全面评估模型性能，收集所有预测结果
    
    返回:
        all_labels: 真实标签
        all_preds: 预测标签
        all_probs: 预测概率
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    绘制混淆矩阵热力图
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.figure.colorbar(im, ax=ax, shrink=0.8)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names,
           yticklabels=class_names,
           title='Confusion Matrix (Normalized)',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格中显示数值
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {save_path}")
    plt.show()
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    绘制每个类别的精确率、召回率、F1分数柱状图
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#4CAF50', edgecolor='white')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#2196F3', edgecolor='white')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#FF9800', edgecolor='white')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Precision, Recall, and F1-Score', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"各类别指标图已保存至: {save_path}")
    plt.show()
    plt.close()

    return precision, recall, f1


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """
    绘制各类别的ROC曲线
    """
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))
    micro_avg_fpr = np.linspace(0, 1, 100)
    micro_avg_tpr = 0

    for i, color in enumerate(colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    # 绘制对角线
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves for Each Class (One-vs-Rest)', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存至: {save_path}")
    plt.show()
    plt.close()


def visualize_predictions(model, test_loader, class_names, device=DEVICE,
                          num_images=20, save_path=None):
    """
    可视化模型的预测结果（展示正确和错误预测）
    """
    model.eval()
    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)

            images_list.extend(images.cpu())
            labels_list.extend(labels.numpy())
            preds_list.extend(preds.cpu().numpy())

            if len(images_list) >= num_images:
                break

    # 反归一化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(min(num_images, len(images_list))):
        img = images_list[i]
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        true_label = class_names[labels_list[i]]
        pred_label = class_names[preds_list[i]]
        is_correct = labels_list[i] == preds_list[i]

        axes[i].imshow(img)
        color = 'green' if is_correct else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {pred_label}',
                          color=color, fontsize=9)
        axes[i].axis('off')

    plt.suptitle('Model Predictions (Green=Correct, Red=Wrong)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"预测结果图已保存至: {save_path}")
    plt.show()
    plt.close()


def analyze_errors(model, test_loader, class_names, device=DEVICE,
                   num_errors=20, save_path=None):
    """
    错误案例分析：展示模型预测错误的典型样本
    """
    model.eval()
    error_images = []
    error_true = []
    error_pred = []
    error_conf = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(1)

            # 找到错误预测的样本
            mask = preds != labels
            if mask.any():
                error_images.extend(images[mask].cpu())
                error_true.extend(labels[mask].cpu().numpy())
                error_pred.extend(preds[mask].cpu().numpy())
                error_conf.extend(confidences[mask].cpu().numpy())

            if len(error_images) >= num_errors:
                break

    if len(error_images) == 0:
        print("没有找到错误预测！")
        return

    # 反归一化
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    num_show = min(num_errors, len(error_images))
    rows = (num_show + 4) // 5
    fig, axes = plt.subplots(rows, 5, figsize=(15, 3 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    for i in range(num_show):
        img = error_images[i]
        img = img * std + mean
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0).numpy()

        axes[i].imshow(img)
        axes[i].set_title(f'True: {class_names[error_true[i]]}\n'
                          f'Pred: {class_names[error_pred[i]]}\n'
                          f'Conf: {error_conf[i]*100:.1f}%',
                          color='red', fontsize=9)
        axes[i].axis('off')

    for i in range(num_show, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Error Case Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"错误案例分析图已保存至: {save_path}")
    plt.show()
    plt.close()


def compute_sensitivity_specificity(y_true, y_pred, num_classes):
    """
    计算每个类别的敏感性和特异性
    敏感性 = TP / (TP + FN)  （即召回率）
    特异性 = TN / (TN + FP)
    """
    cm = confusion_matrix(y_true, y_pred)
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity[i] = tn / (tn + fp) if (tn + fp) > 0 else 0

    return sensitivity, specificity


def print_classification_report(y_true, y_pred, class_names):
    """
    打印完整分类报告
    包含：准确率、敏感性、特异性、召回率、精确率、F1值
    """
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    # 计算每个类别的精确率、召回率、F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    # 计算敏感性和特异性
    sensitivity, specificity = compute_sensitivity_specificity(
        y_true, y_pred, len(class_names)
    )

    # 计算总体指标
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    accuracy = np.mean(y_true == y_pred) * 100
    avg_sensitivity = sensitivity.mean()
    avg_specificity = specificity.mean()

    print(f"\n各类别敏感性 (Sensitivity/Recall):")
    for i, name in enumerate(class_names):
        print(f"  {name:<12s}: {sensitivity[i]:.4f}")

    print(f"\n各类别特异性 (Specificity):")
    for i, name in enumerate(class_names):
        print(f"  {name:<12s}: {specificity[i]:.4f}")

    print(f"\n总体指标:")
    print(f"  准确率 (Accuracy):            {accuracy:.2f}%")
    print(f"  宏平均精确率 (Macro Precision): {macro_precision:.4f}")
    print(f"  宏平均召回率 (Macro Recall):    {macro_recall:.4f}")
    print(f"  宏平均F1分数 (Macro F1-Score):  {macro_f1:.4f}")
    print(f"  平均敏感性 (Avg Sensitivity):   {avg_sensitivity:.4f}")
    print(f"  平均特异性 (Avg Specificity):   {avg_specificity:.4f}")
    print("=" * 60)

    return accuracy, macro_precision, macro_recall, macro_f1, sensitivity, specificity


def plot_sensitivity_specificity(y_true, y_pred, class_names, save_path=None):
    """
    绘制敏感性和特异性对比图
    满足老师要求的敏感性和特异性指标可视化
    """
    sensitivity, specificity = compute_sensitivity_specificity(
        y_true, y_pred, len(class_names)
    )

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, sensitivity, width,
                   label='Sensitivity (Recall)', color='#4CAF50', edgecolor='white')
    bars2 = ax.bar(x + width/2, specificity, width,
                   label='Specificity', color='#FF9800', edgecolor='white')

    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Sensitivity and Specificity by Class',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"敏感性/特异性图已保存至: {save_path}")
    plt.show()
    plt.close()

    return sensitivity, specificity


if __name__ == "__main__":
    from models import ClassicCNN, resnet18
    from data_utils import get_data_loaders
    from train import train_model

    # 获取数据
    train_loader, test_loader, _, _ = get_data_loaders()

    # 加载训练好的模型进行评估
    model = ClassicCNN(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load("../models/ClassicCNN_best.pth")['model_state_dict'])

    # 评估
    y_true, y_pred, y_prob = evaluate_model(model, test_loader)
    print_classification_report(y_true, y_pred, CLASS_NAMES)
    plot_confusion_matrix(y_true, y_pred, CLASS_NAMES,
                          save_path="../figures/confusion_matrix.png")
