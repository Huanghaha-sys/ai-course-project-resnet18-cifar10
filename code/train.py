"""
训练模块
包含模型训练、验证、保存功能
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
import time
from config import *


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    训练一个epoch
    
    参数:
        model: 神经网络模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备(CPU/GPU)
    
    返回:
        avg_loss: 平均损失
        accuracy: 训练准确率
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(model, test_loader, criterion, device):
    """
    在测试集上验证模型性能
    
    返回:
        avg_loss: 平均损失
        accuracy: 测试准确率
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Validating", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, test_loader, model_name="model",
                num_epochs=NUM_EPOCHS, lr=LEARNING_RATE, device=DEVICE):
    """
    完整训练流程
    
    参数:
        model: 待训练的模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        model_name: 模型名称（用于保存）
        num_epochs: 训练轮数
        lr: 学习率
        device: 计算设备
    
    返回:
        history: 训练历史记录
    """
    print(f"\n{'='*60}")
    print(f"开始训练: {model_name}")
    print(f"设备: {device}")
    print(f"学习率: {lr}")
    print(f"训练轮数: {num_epochs}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*60}\n")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # 学习率衰减策略：余弦退火（更平滑，更接近论文标准做法）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # 训练历史
    history = {
        'model_name': model_name,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epochs': []
    }

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        # 验证
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        # 更新学习率
        scheduler.step()

        # 记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epochs'].append(epoch + 1)

        epoch_time = time.time() - epoch_start

        print(f"Epoch [{epoch+1:3d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = os.path.join(MODEL_SAVE_DIR, f"{model_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_path)
            print(f"  -> 最佳模型已保存 (Acc: {val_acc:.2f}%)")

    total_time = time.time() - start_time
    print(f"\n训练完成！总耗时: {total_time/60:.2f} 分钟")
    print(f"最佳验证准确率: {best_acc:.2f}%")

    # 保存训练历史
    history['best_acc'] = best_acc
    history['total_time'] = total_time
    history_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"训练历史已保存至: {history_path}")

    return history


if __name__ == "__main__":
    from models import ClassicCNN, resnet18
    from data_utils import get_data_loaders

    # 测试训练
    train_loader, test_loader, _, _ = get_data_loaders()

    # 训练ClassicCNN
    model = ClassicCNN(num_classes=10).to(DEVICE)
    history = train_model(model, train_loader, test_loader,
                          model_name="ClassicCNN", num_epochs=2)
