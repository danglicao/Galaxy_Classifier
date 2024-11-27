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
import os
import torch.nn as nn
from tqdm import tqdm

class GalaxyDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        self.h5_file_path = h5_file_path
        self.transform = transform
        self.h5_file = h5py.File(h5_file_path, 'r')

        # 加载标签
        self.labels = self.h5_file['labels'][:]
        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img_bytes = self.h5_file['image_data'][idx]
        label = self.h5_file['labels'][idx]

        # 转换图像为RGB
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # 应用变换
        if self.transform:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)
        return img, label

    def __del__(self):
        if self.h5_file is not None:
            self.h5_file.close()

def load_test_data(h5_file_path, batch_size, model_name):
    if model_name == 'vit_l_16':
        transform = transforms.Compose([
            # resnet
            # transforms.Resize((224, 224)),
            #vit
            transforms.Resize((512,512)),
            # convnext
            # transforms.Resize((232, 232)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
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
        total_batches = len(test_loader)
        for images, labels in tqdm(test_loader, total = total_batches, desc = "Testing Progress"):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        # for images, labels in test_loader:
        #     images = images.to(device)
        #     labels = labels.to(device)
        #     outputs = model(images)
        #     _, predicted = torch.max(outputs.data, 1)
        #     all_preds.extend(predicted.cpu().numpy())
        #     all_labels.extend(labels.cpu().numpy())
        #     # print(predicted)
        #     print(f'finish test {len(all_labels)} samples')

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


def build_model(num_classes=3, model_name=None):
    # model = models.convnext_base(weights='IMAGENET1K_V1')
    # model = models.resnet152(weights = None)
    # model = models.vit_l_16(weights = 'IMAGENET1K_SWAG_E2E_V1')
    # model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    # for param in model.features.parameters():
    #     param.requires_grad = False
    # model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, num_classes)
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, num_classes)
    # model.fc.requires_grad = True
    # num_patches = (input_size // model.patch_size) ** 2
    # pos_embedding = model.encoder.pos_embedding
    # if pos_embedding.shape[1] != num_patches + 1:
    #     cls_token = pos_embedding[:, :1, :]  # 保留CLS token的嵌入
    #     pos_tokens = pos_embedding[:, 1:, :]  # 提取patch嵌入
    #     pos_tokens = nn.functional.interpolate(
    #         pos_tokens.reshape(1, int(pos_tokens.shape[1] ** 0.5), int(pos_tokens.shape[1] ** 0.5),
    #                            -1).permute(0, 3, 1, 2),
    #         size = (input_size // model.patch_size, input_size // model.patch_size),
    #         mode = 'bicubic',
    #         align_corners = False
    #     ).permute(0, 2, 3, 1).reshape(1, -1, pos_embedding.shape[-1])
    #     model.encoder.pos_embedding = nn.Parameter(torch.cat([cls_token, pos_tokens], dim = 1))
    if model_name == 'convnext':
        model = models.convnext_base(weights = 'IMAGENET1K_V1')
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(weights = None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_l_16':
        model = models.vit_l_16(weights = 'IMAGENET1K_SWAG_E2E_V1')
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name == 'resnet152':
        model = models.resnet152(weights = None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif model_name == 'vit_h_14':
        model = models.vit_h_14(weights = None)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError("Invalid model name. Please choose from 'convnext', 'resnet50', 'vit_l_16', 'resnet152', 'vit_h_14'.")

    return model


    return model


def main():
    h5_file_path = 'D:/Ame508_final_data/galaxy_zoo/test_data.h5'
    model_path = 'D:/Ame508_final_data/result/convnext_base/best_model.pth'
    dir_path = 'D:/Ame508_final_data/result/convnext_base'
    batch_size = 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = 'convnext'

    test_loader, num_samples = load_test_data(h5_file_path, batch_size, model_name)

    # 构建模型并加载权重
    model = build_model(num_classes=3, model_name = model_name)
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    print(f"Testing on {num_samples} samples.")
    evaluate_model(model, test_loader, device, dir_path)


if __name__ == '__main__':
    main()