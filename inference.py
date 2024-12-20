import torch
from torchvision import transforms
from PIL import Image
import argparse
import os
import torch.nn as nn
from torchvision import models


def build_model(num_classes = 3, model_name = None):


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
        raise ValueError(
            "Invalid model name. Please choose from 'convnext', 'resnet50', 'vit_l_16', 'resnet152', 'vit_h_14'.")

    return model


def predict_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs.data, 1)

    return predicted.item()

def main():
    parser = argparse.ArgumentParser(description="Galaxy Image Inference")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model.")
    # parser.add_argument("--num_classes", type=int, default=3, help="Number of classes.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_model(num_classes=3)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # 预测图像
    try:
        predicted_label = predict_image(args.image_path, model, device)
        class_names = ['E', 'S', 'SB']
        print(f"Predicted label: {class_names[predicted_label]}")
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == "__main__":
    # main()
    img_path  = 'D:/Ame508_final_data/galaxy_zoo/images_gz2/images/100011.jpg'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_model(num_classes = 3, model_name = 'resnet152')
    model.load_state_dict(torch.load('D:/Ame508_final_data/result/resnet152/resnet152.pth'))
    model.to(device)
    try:
        predicted_label = predict_image(img_path, model, device)
        class_names = ['E', 'S', 'SB']
        print(f"Predicted label: {class_names[predicted_label]}")
    except Exception as e:
        print(f"Error during inference: {e}")