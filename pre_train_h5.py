from idlelib.pyparse import trans
import io
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
import h5py

# 设置随机种子以确保结果可重复
torch.manual_seed(3407)
np.random.seed(3407)
random.seed(3407)


def load_data(h5_file_path, batch_size, num_workers=1):
    # 打开 HDF5 文件，获取标签信息
    with h5py.File(h5_file_path, 'r') as h5_file:
        labels = h5_file['labels'][:]
        num_classes = len(np.unique(labels))
        num_samples = len(labels)

    # 划分训练集和验证集索引
    indices = np.arange(num_samples)
    train_indices, val_indices, _, _ = train_test_split(
        indices,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # 定义数据变换
    train_transform = transforms.Compose([
        #resnet/vit
        transforms.Resize((224, 224)),
        #convnext
        # transforms.Resize((232, 232)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        #resnet/vit
        transforms.Resize((224, 224)),
        #convnext
        # transforms.Resize((232, 232)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225]),
    ])

    # 创建 Dataset 和 DataLoader
    train_dataset = GalaxyDataset(
        h5_file_path=h5_file_path,
        indices=train_indices,
        transform=train_transform
    )

    val_dataset = GalaxyDataset(
        h5_file_path=h5_file_path,
        indices=val_indices,
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
    def __init__(self, h5_file_path, indices, transform=None):
        self.h5_file_path = h5_file_path
        self.indices = indices
        self.transform = transform
        self.h5_file = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_file_path, 'r')

        actual_idx = self.indices[idx]
        img_bytes = self.h5_file['image_data'][actual_idx]
        label = self.h5_file['labels'][actual_idx]

        # 将字节流转换为 PIL 图像
        img = Image.open(io.BytesIO(img_bytes))
        img = img.convert('RGB')

        # 应用数据变换
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __del__(self):
        # 关闭 HDF5 文件
        if self.h5_file is not None:
            self.h5_file.close()



def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device='cuda', patience=7, start_epoch=0, history=None):
    if history is None:
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    model.to(device)
    early_stopping = EarlyStopping(patience=patience, verbose=True, path='D:/Ame508_final_data/result/new_store/best_model.pth')

    for epoch in range(start_epoch, num_epochs):
        # Training Phase
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
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        if scheduler:
            scheduler.step()

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation Phase
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

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, scheduler, history)

        # Early Stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training stopped.")
            break

    model.load_state_dict(torch.load('best_model.pth'))
    return model, history






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
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 定义类别名称
    class_names = ['E', 'S', 'SB']  # 根据您的标签映射

    # 生成分类报告
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    with open("D:/Ame508_final_data/result/new_store/classification_report.txt",
              "w") as f:
        f.write(report)

    # 绘制混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('D:/Ame508_final_data/result/new_store/confusion_matrix.png')
    plt.show()

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=10, delta=0, verbose=False, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            verbose (bool): If True, prints a message for each validation loss improvement.
            path (str): Path for the checkpoint to save the model.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved. Saving model to {self.path}.")
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True


def save_checkpoint(epoch, model, optimizer, scheduler, history, path='D:/Ame508_final_data/result/new_store/checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': history
    }
    torch.save(checkpoint, path)

def load_checkpoint(path, model, optimizer, scheduler):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']

    return model, optimizer, scheduler, start_epoch, history

def build_model(num_classes):

    # model = models.resnet34(weights = 'IMAGENET1K_V1')
    # model = models.resnet34(weights = None)
    # model = models.vit_b_32(weights = 'IMAGENET1K_V1')
    # model = models.resnet152(weights = 'IMAGENET1K_V2')
    model = models.convnext_base(weights = 'IMAGENET1K_V1')
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    # Transformer
    # for name, param in model.named_parameters():
    #     if "heads" not in name:
    #         param.requires_grad = False
    # model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    # for name, param in model.named_parameters():
    #     if "encoder_layer_11" in name:
    #         param.requires_grad = True

    # CNN
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.fc.requires_grad = True

    # convnext
    for param in model.features.parameters():
      param.requires_grad = False

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    model.classifier[2].requires_grad = True
    for param in model.features[3].parameters():
        param.requires_grad = True
    return model


def main():
    # 参数设置
    h5_file_path = 'D:/Ame508_final_data/galaxy_zoo/train_data.h5'
    batch_size = 32
    num_workers = 4
    num_epochs = 1000
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    train_loader, val_loader, num_classes = load_data(
        h5_file_path,
        batch_size,
        num_workers
    )

    # 构建模型
    model = build_model(num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.AdamW(model.heads.parameters(), lr=learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr = learning_rate)
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

    model_save_path = 'D:/Ame508_final_data/result/new_store/galaxy_classifier.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_history(history)
    evaluate_model(model, val_loader, device)

def main_with_check_point():
    h5_file_path = 'D:/Ame508_final_data/galaxy_zoo/train_data.h5'
    batch_size = 32
    num_workers = 4
    num_epochs = 1000
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()

    # 加载数据
    train_loader, val_loader, num_classes = load_data(
        h5_file_path,
        batch_size,
        num_workers
    )
    path = 'D:/Ame508_final_data/result/new_store/checkpoint.pth'
    optimizer = optim.AdamW(build_model(3).parameters(), lr = learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optim.AdamW(build_model(3).parameters(), lr = learning_rate), step_size=7, gamma=0.1)
    model, optimizer, scheduler, start_epoch, history = load_checkpoint(path = path, model = build_model(3), optimizer = optimizer, scheduler = scheduler)

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

    model_save_path = 'D:/Ame508_final_data/result/new_store/galaxy_classifier.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    plot_history(history)
    evaluate_model(model, val_loader, device)



if __name__ == '__main__':
    # main()
    main_with_check_point()
    # model = build_model(3)
    # print(model)
    # for name, param in model.named_parameters():
    #     print(f"{name}: {'可训练' if param.requires_grad else '冻结'}")
