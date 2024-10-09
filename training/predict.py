import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.my_models import SiameseUNetWithResnet50Encoder  # 导入您的模型
from utils.data_loading import preprocess  # 导入您的预处理函数
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics import F1Score, JaccardIndex

def load_model(model_path):
    model = SiameseUNetWithResnet50Encoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_image(model, pre_image, post_image, device):
    model.to(device)
    pre_image = pre_image.unsqueeze(0).to(device)
    post_image = post_image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(pre_image, post_image)
    return output.argmax(dim=1).squeeze().cpu().numpy()

def visualize_prediction(image, mask, prediction):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(mask)
    ax2.set_title('Ground Truth')
    ax3.imshow(prediction)
    ax3.set_title('Prediction')
    plt.show()

def calculate_metrics(prediction, ground_truth):
    prediction_tensor = torch.from_numpy(prediction).unsqueeze(0)
    ground_truth_tensor = torch.from_numpy(ground_truth).unsqueeze(0)
    
    dice = multiclass_dice_coeff(
        F.one_hot(prediction_tensor, 5).permute(0, 3, 1, 2).float(),
        F.one_hot(ground_truth_tensor, 5).permute(0, 3, 1, 2).float(),
        reduce_batch_first=False
    )
    
    f1_score = F1Score(task='multiclass', num_classes=5, average='macro')
    f1 = f1_score(prediction_tensor, ground_truth_tensor)
    
    iou_score = JaccardIndex(task="multiclass", num_classes=5)
    iou = iou_score(prediction_tensor, ground_truth_tensor)
    
    return dice.item(), f1.item(), iou.item()

def main():
    model_path = 'path/to/your/trained/model.pth'
    pre_image_path = 'path/to/pre_disaster_image.png'
    post_image_path = 'path/to/post_disaster_image.png'
    mask_path = 'path/to/ground_truth_mask.png'  # 如果有的话
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    
    pre_image = Image.open(pre_image_path)
    post_image = Image.open(post_image_path)
    mask = Image.open(mask_path) if mask_path else None
    
    pre_image_tensor = preprocess(pre_image).to(device)
    post_image_tensor = preprocess(post_image).to(device)
    
    prediction = predict_image(model, pre_image_tensor, post_image_tensor, device)
    
    if mask is not None:
        mask_np = np.array(mask)
        visualize_prediction(np.array(post_image), mask_np, prediction)
        dice, f1, iou = calculate_metrics(prediction, mask_np)
        print(f'Dice Score: {dice:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'IoU: {iou:.4f}')
    else:
        plt.imshow(prediction)
        plt.title('Prediction')
        plt.show()

if __name__ == '__main__':
    main()
