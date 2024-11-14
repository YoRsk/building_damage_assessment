import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.my_models import SiameseUNetWithResnet50Encoder, SiamUNetConCVgg19  # 导入您的模型
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics import F1Score, JaccardIndex, AUROC, Precision, Recall
import torch.nn.functional as F
import matplotlib.colors as mcolors
import os
import sys
from pathlib import Path
from tqdm import tqdm
import argparse  # 添加到文件开头

def load_model(model_path):
    model = SiameseUNetWithResnet50Encoder()
    # model = SiamUNetConCVgg19()
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model

def preprocess_image(image, size=None):
    """
    预处理图像：调整大小，转换为RGB，标准化

    Args:
        image: PIL Image 格式的图像
        size: 目标大小，如果为None则调整为32的倍数
    """
    # 确保是RGB模式
    if image.mode == 'RGBA':
        # 转换RGBA为RGB，丢弃alpha通道
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        raise ValueError(f"Unsupported image mode: {image.mode}")
    
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

def visualize_prediction(image, mask, prediction, show_original_unclassified=False):
    # 打印每个类别的像素数量和百分比
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    print("\n类别统计:")
    class_names = ['Unclassified', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    for value, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"{class_names[value]}: {count} pixels ({percentage:.2f}%)")
    # 定义颜色映射
    colors = ['black', 'blue', 'green', 'yellow', 'red']
    n_classes = 5
    cmap = mcolors.ListedColormap(colors[:n_classes])
    
    # 创建规范化对象
    bounds = np.arange(n_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 增加子图之间的间距
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # 设置全局显示参数
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.resample'] = False

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    # 显示原始图像
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')
    # 显示Ground Truth（如果有）
    if mask is not None:
        im2 = ax2.imshow(mask,
                        cmap=cmap,
                        norm=norm)
        ax2.set_title('Ground Truth')
    else:
        ax2.set_visible(False)  # 如果没有 mask，隐藏中间的子图
    ax2.axis('off')

    # 修改显示预测结果的部分
    if show_original_unclassified:
        # 创建一个带有 alpha 通道的预测图像
        prediction_colored = np.zeros((*prediction.shape, 4))
        
        # 设置颜色映射（RGBA格式）
        color_map = {
            0: [0, 0, 0, 0],      # 未分类 - 完全透明
            1: [0, 0, 1, 1],      # 蓝色不透明
            2: [0, 1, 0, 1],      # 绿色不透明
            3: [1, 1, 0, 1],      # 黄色不透明
            4: [1, 0, 0, 1],      # 红色不透明
        }
        
        # 为每个类别设置颜色
        for class_idx, color in color_map.items():
            mask = prediction == class_idx
            prediction_colored[mask] = color
            
        # 显示原始图像
        ax3.imshow(image)
        # 叠加预测结果
        overlay = ax3.imshow(prediction_colored)
        
        # 修改颜色条显示
        cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])
        cbar = plt.colorbar(
            overlay,
            cax=cbar_ax,
            orientation='horizontal'
        )
    else:
        # 原来的显示方式
        im3 = ax3.imshow(prediction, cmap=cmap, norm=norm)
    
    ax3.set_title('Prediction')
    ax3.axis('off')

    # 添加颜色条
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.02])  # [left, bottom, width, height]
    cbar = plt.colorbar(
        ax3.imshow(prediction, cmap=cmap, norm=norm),
        cax=cbar_ax,
        orientation='horizontal'
    )
    cbar.set_ticks(bounds[:-1] + 0.5)
    # 设置颜色条标签
    cbar.set_ticklabels(['Unclassified', 'No Damage', 'Minor Damage',
                        'Major Damage', 'Destroyed'])

    # 确保像素级显示
    for ax in [ax2, ax3]:
        if ax.get_visible():
            ax.set_aspect('equal', adjustable='box')

    # 添加事件处理以确保缩放时保持设置
    def on_draw(event):
        for ax in [ax2, ax3]:
            if ax.get_visible():
                ax.images[0].set_interpolation('nearest')

    fig.canvas.mpl_connect('draw_event', on_draw)

    plt.show(block=True)

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
    
    # 直接使用 float32
    output = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    # 计算总窗口数
    y_steps = (height - overlap) // stride
    x_steps = (width - overlap) // stride
    total_steps = y_steps * x_steps
    
    with tqdm(total=total_steps, desc='Processing windows') as pbar:
        for y in range(0, height - overlap, stride):
            for x in range(0, width - overlap, stride):
                # 确定当前窗口的范围
                end_y = min(y + window_size, height)
                end_x = min(x + window_size, width)
                y1 = max(0, y)
                x1 = max(0, x)
                
                # 获取当前窗口的实际大小
                current_height = end_y - y1
                current_width = end_x - x1
                
                # 裁剪图像
                pre_window = pre_image.crop((x1, y1, end_x, end_y))
                post_window = post_image.crop((x1, y1, end_x, end_y))
                
                # 预测当前窗口，并指定目标大小
                pred = predict_image(model, pre_window, post_window, device)
                
                # 如果预测结果大小不匹配，需要调整大小
                if pred.shape != (current_height, current_width):
                    pred = transforms.Resize((current_height, current_width))(
                        torch.from_numpy(pred).unsqueeze(0)
                    ).squeeze(0).numpy()
                
                # 累加预测结果
                output[y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1
                
                pbar.update(1)
    
    # 取平均并四舍五入
    output = np.round(output / counts).astype(np.uint8)
    
    return output

def main():
    # 添加命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-original', action='store_true',
                      help='Show original image for unclassified areas')
    args = parser.parse_args()
    
    # model_path = './checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch49.pth'
    # model_path = './checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch52.pth'
    model_path = './checkpoints/best0921.pth'
    # # xbd
    # img_home_path = "C:/Users/xiao/peng/xbd/Dataset/Validation"
    # pre_image_path = img_home_path + "/Pre/Image512/"+ "hurricane-michael_00000400_pre_disaster.png"
    # post_image_path = img_home_path + "/Post/Image512/"+ "hurricane-michael_00000400_post_disaster.png"
    # mask_path = img_home_path + "/Post/Label512/"+ "hurricane-michael_00000400_post_disaster.png"

    # #my dataset
    # pre_image_path = './images/20210709_073742_79_2431_3B_Visual_clip.tif'
    # post_image_path = './images/20220709_072527_82_242b_3B_Visual_clip.tif'
    # mask_path = ''

    #my dataset2
    pre_image_path = './images/before_Zhovteneyvi.jpeg'
    post_image_path = './images/after_Zhovteneyvi.jpeg'
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
        visualize_prediction(np.array(post_image), mask_np, prediction, args.show_original)
        dice, f1, iou, precision, recall = calculate_metrics(prediction, mask_np)
        print(f'Dice Score: {dice:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'IoU: {iou:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
    else:
        visualize_prediction(np.array(post_image), None, prediction, args.show_original)

if __name__ == '__main__':
    main()
