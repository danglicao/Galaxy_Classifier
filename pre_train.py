from idlelib.pyparse import trans

import numpy as np
import torch
import torchvision
import pandas as pd
import matplotlib.pyplot as plt
import math
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os
import sklearn
from sklearn.model_selection import train_test_split
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import random

# 设置随机种子以确保结果可重复
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)

def load_data(data_path, csv_file, img_dir, batch_size, num_workers=1):

    csv_path = os.path.join(data_path, csv_file)
    img_dir = os.path.join(data_path, img_dir)
    data = pd.read_csv(csv_path)

    # 标签映射
    label_mapping = {'E': 0, 'S': 1, 'SB': 2}
    data['label'] = data['galaxy_type'].map(label_mapping)
    num_classes = len(label_mapping)

    # 划分数据集
    train_df, val_df = train_test_split(
        data,
        test_size=0.2,
        stratify=data['label'],
        random_state=42
    )

    # 定义数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]),

    ])

    # 创建Dataset和DataLoader
    train_dataset = GalaxyDataset(
        dataframe=train_df,
        img_dir=img_dir,
        transform=train_transform
    )

    val_dataset = GalaxyDataset(
        dataframe=val_df,
        img_dir=img_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, num_classes

class GalaxyDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # 获取图像文件名和标签
        img_name = str(self.dataframe.iloc[idx]['asset_id']) + '.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        label = self.dataframe.iloc[idx]['label']
        label = torch.tensor(label, dtype=torch.long)

        # 打开图像
        image = Image.open(img_path).convert('RGB')

        # 应用数据变换
        if self.transform:
            image = self.transform(image)

        return image, label

def build_model(num_classes):

    # model = models.resnet34(weights = 'IMAGENET1K_V1')
    # model = models.resnet34(weights = None)
    model = models.vit_b_16(weights = 'IMAGENET1K_V1')
    # for name, param in model.parameters():
    #     param.requires_grad = False

    # Transformer
    for name, param in model.named_parameters():
        if "heads" not in name:
            param.requires_grad = False
    # for name, param in model.named_parameters():
    #     if "encoder_layer_11" in name:  # 假设有12层，索引从0开始
    #         param.requires_grad = True

    # CNN
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.fc.requires_grad = True

    return model

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler = None, num_epochs=10, device='cuda'):

    model.to(device)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # 计算训练准确率
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        if scheduler:
            scheduler.step()

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history


def main():
    # 参数设置
    data_path = 'D:/Ame508_final_data/galaxy_zoo'
    csv_file = 'subset_labels.csv'
    img_dir = 'images_gz2/images'
    batch_size = 32
    num_workers = 1
    num_epochs = 20
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader, num_classes = load_data(
        data_path,
        csv_file,
        img_dir,
        batch_size,
        num_workers
    )

    # 构建模型
    model = build_model(num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # for CNN
    # optimizer = optim.AdamW(model.fc.parameters(), lr=learning_rate)
    # for Transformer
    optimizer = optim.AdamW(model.heads.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # 训练模型
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs,
        device
    )

    # 保存模型（可选）
    model_save_path = 'galaxy_classifier.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # 绘制训练曲线（可选）
    plot_history(history)
    evaluate_model(model, val_loader, device)

def plot_history(history):

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def evaluate_model(model, val_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=['E', 'S', 'SB'])
    print(report)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['E', 'S', 'SB'], yticklabels=['E', 'S', 'SB'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
    # model = build_model(3)
    # # print(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {'可训练' if param.requires_grad else '冻结'}")
