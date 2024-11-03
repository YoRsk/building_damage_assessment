import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.my_models import SiameseUNetWithResnet50Encoder  # 导入您的模型
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics import F1Score, JaccardIndex, AUROC, Precision, Recall
import torch.nn.functional as F
import matplotlib.colors as mcolors
import os
import sys
from pathlib import Path
from tqdm import tqdm

def load_model(model_path):
    model = SiameseUNetWithResnet50Encoder()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess_image(image, size=None):
    if size is None:
        # 确保宽高是32的倍数
        w, h = image.size
        w = ((w // 32) + (1 if w % 32 != 0 else 0)) * 32
        h = ((h // 32) + (1 if h % 32 != 0 else 0)) * 32
        size = (w, h)
    
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(image)

def predict_image(model, pre_image, post_image, device):
    model.to(device)
    model.eval()      # 确保模型在评估模式
    
    # 预处理
    pre_image = preprocess_image(pre_image)
    post_image = preprocess_image(post_image)
    
    # 添加batch维度并移到设备
    pre_image = pre_image.unsqueeze(0).to(device)
    post_image = post_image.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(pre_image, post_image)
    #.squeeze(): 由于输入只有一张图片，batch_size 为 1，这步操作去除了 batch 维度。结果形状变为 (height, width)。
    # argmax: get f(x)最大值
    return output.argmax(dim=1).squeeze().cpu().numpy()

def visualize_prediction(image, mask, prediction):
    # 定义颜色映射
    colors = ['black', 'blue', 'green', 'yellow', 'red']
    n_classes = 5
    cmap = mcolors.ListedColormap(colors[:n_classes])
    
    # 创建规范化对象
    bounds = np.arange(n_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    if mask is not None:
        ax2.imshow(mask, cmap=cmap, norm=norm)
        ax2.set_title('Ground Truth')
    else:
        ax2.set_visible(False)  # 如果没有 mask，隐藏中间的子图
    ax2.axis('off')
    
    ax3.imshow(prediction, cmap=cmap, norm=norm)
    ax3.set_title('Prediction')
    ax3.axis('off')
    
    # 添加颜色条
    cbar = fig.colorbar(ax3.imshow(prediction, cmap=cmap, norm=norm), ax=[ax2, ax3], orientation='horizontal', aspect=30, pad=0.08)
    cbar.set_ticks(bounds[:-1] + 0.5)
    cbar.set_ticklabels(['Unclassified', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed'])
    
    plt.tight_layout()
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
    
    precision_score = Precision(task="multiclass", num_classes=5, average='macro')
    precision = precision_score(prediction_tensor, ground_truth_tensor)
    
    recall_score = Recall(task="multiclass", num_classes=5, average='macro')
    recall = recall_score(prediction_tensor, ground_truth_tensor)
    
    # auroc = AUROC(task="multiclass", num_classes=5)
    # pred_probs = F.one_hot(torch.from_numpy(prediction).long(), 5).float()
    # auc_roc = auroc(pred_probs, ground_truth_tensor)
    
    return dice.item(), f1.item(), iou.item(), precision.item(), recall.item()

def predict_with_sliding_window(model, pre_image, post_image, window_size=1024, overlap=32, device='cuda'):
    """
    使用滑动窗口方式预测大图像
    
    Args:
        model: 模型
        pre_image: PIL Image 格式的灾前图像
        post_image: PIL Image 格式的灾后图像
        window_size: 窗口大小
        overlap: 重叠区域大小
        device: 设备
    """
    model.to(device)
    model.eval()
    
    # 获取图像尺寸
    width, height = pre_image.size
    
    # 计算步长
    stride = window_size - overlap
    
    # 创建输出数组
    output = np.zeros((height, width), dtype=np.uint8)
    counts = np.zeros((height, width), dtype=np.uint8)
    
    # 计算总窗口数
    y_steps = (height - overlap) // stride
    x_steps = (width - overlap) // stride
    total_steps = y_steps * x_steps
    
    # 使用tqdm创建进度条
    with tqdm(total=total_steps, desc='Processing windows') as pbar:
        for y in range(0, height - overlap, stride):
            for x in range(0, width - overlap, stride):
                # 确定当前窗口的范围
                end_y = min(y + window_size, height)
                end_x = min(x + window_size, width)
                y1 = max(0, y)
                x1 = max(0, x)
                
                # 裁剪图像
                pre_window = pre_image.crop((x1, y1, end_x, end_y))
                post_window = post_image.crop((x1, y1, end_x, end_y))
                
                # 预测当前窗口
                pred = predict_image(model, pre_window, post_window, device)
                
                # 将预测结果写入输出数组
                output[y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1
                
                # 更新进度条
                pbar.update(1)
    
    # 取平均值
    output = output / counts
    
    # 返回最终预测结果
    return output.astype(np.uint8)

def main():
    model_path = './checkpoints/best0921.pth'
    pre_image_path = './images/20210709_073742_79_2431_3B_Visual_clip.tif'
    post_image_path = './images/20220709_072527_82_242b_3B_Visual_clip.tif'
    mask_path = '' 
    # mask_path = 'path/to/ground_truth_mask.tif'  # 如果有的话
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    model.to(device)
    
    pre_image = Image.open(pre_image_path)
    post_image = Image.open(post_image_path)
    mask = Image.open(mask_path) if mask_path else None
    
    # 对于大图像使用滑动窗口
    if pre_image.size[0] > 1024 or pre_image.size[1] > 1024:
        prediction = predict_with_sliding_window(
            model, pre_image, post_image, 
            window_size=1024, overlap=32, device=device
        )
    else:
        prediction = predict_image(model, pre_image, post_image, device)
    
    if mask is not None:
        mask_np = np.array(mask)
        visualize_prediction(np.array(post_image), mask_np, prediction)
        dice, f1, iou, precision, recall, auc_roc = calculate_metrics(prediction, mask_np)
        print(f'Dice Score: {dice:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'IoU: {iou:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'AUC-ROC: {auc_roc:.4f}')
    else:
        visualize_prediction(np.array(post_image), None, prediction)

if __name__ == '__main__':
    main()
