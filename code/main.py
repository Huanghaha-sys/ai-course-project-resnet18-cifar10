"""
主程序入口 - v5最终执行版
核心原则：复杂度集中在"原理讲解 + 可视化"，不加在实验数量
结构：主线(ResNet-18)极深 + 副线(CNN对照)极轻 + Feature Map可视化很强
"""
import torch
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import *
from models import ClassicCNN, resnet18
from data_utils import (get_data_loaders, visualize_dataset_samples,
                        print_dataset_info, plot_class_distribution,
                        plot_pixel_distribution, plot_preprocessing_comparison)
from train import train_model
from evaluate import (evaluate_model, print_classification_report,
                      plot_confusion_matrix, plot_per_class_metrics,
                      plot_roc_curves, visualize_predictions, analyze_errors,
                      plot_training_curves)
from feature_map import compare_feature_maps
from fruit_transfer import run_fruit_experiment


def main():
    print("\n" + "=" * 70)
    print("  CIFAR-10 图像分类：ResNet-18 复现与性能分析")
    print("  ClassicCNN 作为 Baseline 对照")
    print("=" * 70)

    # ==================== 1. 数据准备 ====================
    print("\n【步骤 1/6】加载 CIFAR-10 数据集...")
    train_loader, test_loader, train_dataset, test_dataset = get_data_loaders()
    print_dataset_info(train_dataset, test_dataset)

    # 数据可视化（满足老师"直方图"要求）
    print("\n  -> 生成数据集可视化...")
    visualize_dataset_samples(
        train_dataset,
        save_path=os.path.join(FIGURES_DIR, "cifar10_samples.png")
    )
    plot_class_distribution(
        train_dataset, test_dataset,
        save_path=os.path.join(FIGURES_DIR, "cifar10_class_distribution.png")
    )
    plot_pixel_distribution(
        train_dataset,
        save_path=os.path.join(FIGURES_DIR, "cifar10_pixel_distribution.png")
    )
    # 预处理前后对比图（满足老师"预处理前后对比"要求）
    print("\n  -> 生成预处理对比图...")
    plot_preprocessing_comparison(
        train_dataset,
        save_path=os.path.join(FIGURES_DIR, "cifar10_preprocessing_comparison.png")
    )

    # ==================== 2. 训练两个模型 ====================
    print("\n【步骤 2/6】训练 ClassicCNN（Baseline 对照）...")
    cnn_model = ClassicCNN(num_classes=NUM_CLASSES).to(DEVICE)
    cnn_history = train_model(cnn_model, train_loader, test_loader,
                              model_name="ClassicCNN",
                              num_epochs=NUM_EPOCHS,
                              device=DEVICE)

    print("\n【步骤 3/6】训练 ResNet-18（复现核心）...")
    resnet_model = resnet18(num_classes=NUM_CLASSES).to(DEVICE)
    resnet_history = train_model(resnet_model, train_loader, test_loader,
                                 model_name="ResNet18",
                                 num_epochs=NUM_EPOCHS,
                                 device=DEVICE)

    # ==================== 3. 训练曲线对比 ====================
    print("\n【步骤 4/6】生成训练曲线对比...")
    plot_training_curves([cnn_history, resnet_history],
                         save_path=os.path.join(FIGURES_DIR, "training_curves_comparison.png"))

    # ==================== 4. 评估 ClassicCNN（精简）====================
    print("\n【步骤 5/6】ClassicCNN 评估（Baseline 精简版）...")
    cnn_model.load_state_dict(
        torch.load(os.path.join(MODEL_SAVE_DIR, "ClassicCNN_best.pth"),
                   map_location=DEVICE)['model_state_dict']
    )
    y_true_cnn, y_pred_cnn, y_prob_cnn = evaluate_model(cnn_model, test_loader, DEVICE)
    results_cnn = print_classification_report(y_true_cnn, y_pred_cnn, CLASS_NAMES)
    acc_cnn, prec_cnn, rec_cnn, f1_cnn, sens_cnn, spec_cnn = results_cnn

    # ClassicCNN只做：混淆矩阵
    plot_confusion_matrix(y_true_cnn, y_pred_cnn, CLASS_NAMES,
                          save_path=os.path.join(FIGURES_DIR, "ClassicCNN_confusion_matrix.png"))

    # ==================== 5. 评估 ResNet-18（完整）====================
    print("\n【步骤 6/6】ResNet-18 评估（完整深度分析）...")
    resnet_model.load_state_dict(
        torch.load(os.path.join(MODEL_SAVE_DIR, "ResNet18_best.pth"),
                   map_location=DEVICE)['model_state_dict']
    )
    y_true_rn, y_pred_rn, y_prob_rn = evaluate_model(resnet_model, test_loader, DEVICE)
    results_rn = print_classification_report(y_true_rn, y_pred_rn, CLASS_NAMES)
    acc_rn, prec_rn, rec_rn, f1_rn, sens_rn, spec_rn = results_rn

    # ResNet-18完整评估：混淆矩阵 + 各类别指标 + ROC曲线 + 错误分析
    plot_confusion_matrix(y_true_rn, y_pred_rn, CLASS_NAMES,
                          save_path=os.path.join(FIGURES_DIR, "ResNet18_confusion_matrix.png"))
    plot_per_class_metrics(y_true_rn, y_pred_rn, CLASS_NAMES,
                           save_path=os.path.join(FIGURES_DIR, "ResNet18_per_class_metrics.png"))
    plot_roc_curves(y_true_rn, y_prob_rn, CLASS_NAMES,
                    save_path=os.path.join(FIGURES_DIR, "ResNet18_roc_curves.png"))
    # 预测结果可视化（正确/错误样本展示）
    visualize_predictions(resnet_model, test_loader, CLASS_NAMES,
                          save_path=os.path.join(FIGURES_DIR, "ResNet18_predictions.png"))
    analyze_errors(resnet_model, test_loader, CLASS_NAMES,
                   save_path=os.path.join(FIGURES_DIR, "ResNet18_error_analysis.png"))

    # ==================== 6. Feature Map 可视化（隐藏加分点）====================
    print("\n" + "=" * 70)
    print("  【Feature Map 可视化 - 隐藏加分点】")
    print("  直观展示 CNN vs ResNet 特征提取差异")
    print("=" * 70)
    compare_feature_maps(cnn_model, resnet_model, test_loader,
                         save_dir=FIGURES_DIR, device=DEVICE)

    # ==================== 7. 自构水果数据集迁移学习（加分项）====================
    print("\n" + "=" * 70)
    print("  【加分项：自构水果数据集 + 迁移学习】")
    print("=" * 70)
    print("  如果已准备好水果照片，将自动运行此实验")
    print("  数据集路径: ./fruit_dataset/")
    print("  如果没有准备，此步骤将跳过，不影响主体实验")
    print("=" * 70)

    fruit_result = run_fruit_experiment(
        data_dir='./fruit_dataset',
        figures_dir=FIGURES_DIR,
        device=DEVICE
    )

    # ==================== 8. 最终汇总 ====================
    print("\n" + "=" * 70)
    print("  实验完成！结果汇总")
    print("=" * 70)

    print(f"\n{'='*60}")
    print("  定量评估结果对比")
    print(f"{'='*60}")
    print(f"{'模型':<15} {'准确率':<10} {'精确率':<10} {'召回率':<10} {'F1':<10}")
    print("-" * 55)
    print(f"{'ClassicCNN':<15} {acc_cnn/100:<10.4f} {prec_cnn:<10.4f} {rec_cnn:<10.4f} {f1_cnn:<10.4f}")
    print(f"{'ResNet-18':<15} {acc_rn/100:<10.4f} {prec_rn:<10.4f} {rec_rn:<10.4f} {f1_rn:<10.4f}")
    print(f"{'='*60}")

    print("\n生成图表清单（按论文使用顺序）：")
    chart_list = [
        ("cifar10_samples.png", "数据集样本展示"),
        ("cifar10_class_distribution.png", "类别分布直方图"),
        ("cifar10_pixel_distribution.png", "像素值分布图"),
        ("cifar10_preprocessing_comparison.png", "预处理前后对比"),
        ("training_curves_comparison.png", "训练曲线对比"),
        ("ClassicCNN_confusion_matrix.png", "CNN混淆矩阵（精简对照）"),
        ("ResNet18_confusion_matrix.png", "ResNet混淆矩阵（核心结果）"),
        ("ResNet18_per_class_metrics.png", "各类别精确率/召回率/F1"),
        ("ResNet18_roc_curves.png", "ROC曲线与AUC（仅ResNet）"),
        ("ResNet18_predictions.png", "预测结果可视化"),
        ("ResNet18_error_analysis.png", "错误案例分析"),
        ("featuremap_input_images.png", "特征可视化输入图像"),
        ("featuremap_cnn_vs_resnet_shallow.png", "浅层特征对比（隐藏加分）"),
        ("featuremap_cnn_vs_resnet_deep.png", "深层特征对比（隐藏加分）"),
        ("featuremap_multilayer_evolution.png", "多层特征演化（隐藏加分）"),
    ]

    if fruit_result is not None:
        chart_list.append(("fruit_predictions.png", "水果数据集预测结果（加分项）"))

    for i, (fname, desc) in enumerate(chart_list, 1):
        print(f"  {i:2d}. {fname:<45s} - {desc}")

    print(f"\n{'='*60}")
    print(f"代码生成图片总数: {len(chart_list)}张")
    print("论文中需自行绘制的图（PPT/Visio）：")
    print("  1. 实验流程图")
    print("  2. ResNet-18网络结构图")
    print("  3. 残差块（BasicBlock）细节图")
    print(f"\n提示：请在本地运行时截图CMD窗口，作为代码运行截图")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
