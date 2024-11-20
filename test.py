import io
import numpy as np
import torch
import h5py
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from pre_train_h5 import GalaxyDataset
import os

def load_test_data(h5_file_path, batch_size):
    transform = transforms.Compose([
        # resnet/vit
        transforms.Resize((224, 224)),
        # convnext
        # transforms.Resize((232, 232)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_dataset = GalaxyDataset(h5_file_path=h5_file_path, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, len(test_dataset)

def evaluate_model(model, test_loader, device, dir_path):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    class_names = ['E', 'S', 'SB']


    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)
    with open(os.path.join(dir_path,'classification_report_test.txt'), "w") as f:
        f.write(report)


    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(dir_path,'confusion_matrix_test.png'))
    plt.show()


def build_model(num_classes):
    model = models.convnext_base(weights='IMAGENET1K_V1')
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    return model


def main():
    h5_file_path = 'D:/Ame508_final_data/galaxy_zoo/test_data.h5'
    model_path = 'D:/Ame508_final_data/result/new_store/best_model.pth'
    dir_path = 'D:/Ame508_final_data/result/new_store'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_loader, num_samples = load_test_data(h5_file_path, batch_size)

    # 构建模型并加载权重
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    print(f"Testing on {num_samples} samples.")
    evaluate_model(model, test_loader, device, dir_path)


if __name__ == '__main__':
    main()