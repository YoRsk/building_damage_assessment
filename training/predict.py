import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchmetrics
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
# 在文件开头添加必要的导入
from skimage import measure
import torch.nn.functional as F
from scipy import ndimage
from matplotlib.lines import Line2D  # 添加这一行
Image.MAX_IMAGE_PIXELS = None  # 禁用图像大小限制警告

# 0.75m resolution
# CONFIG = {
#     'SMALL_BUILDING_THRESHOLD': 400,  # Adjusted to four times for 0.75m resolution
#     'WINDOW_SIZE': 256,     # Setting this to 512 will miss many small targets
#     'OVERLAP': 64,         # Increased overlap
#     'CONTEXT_WINDOW': 32,  # Context window size
#     'DAMAGE_THRESHOLD': 0.3, # Damage assessment threshold
# }
# 3m resolution
#previous 44.5
CONFIG = {
    'SMALL_BUILDING_THRESHOLD': 20,   # Pixel threshold for small buildings at 3m resolution
    'WINDOW_SIZE': 512,    # Sliding window size
    #'WINDOW_SIZE': 20000, # means no sliding window and no post-processing
    'OVERLAP': 32,         # Overlap region size
    'CONTEXT_WINDOW': 64,  # Context window size
    'DAMAGE_THRESHOLD': 2,# Damage assessment threshold
}
def load_model(model_path):
    #model = SiameseUNetWithResnet50Encoder()
    model = SiamUNetConCVgg19()
    if model_path:  # 只在路径非空时加载权重
        model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model
class BuildingAwarePredictor:
    """
    用于处理建筑物感知的预测处理器
    
    属性:
        model: 训练好的模型
        device: 计算设备
        config: 配置参数
        damage_classes: 损伤等级字典
    """
    def __init__(self, model, device='cuda', config=CONFIG):
        self.model = model
        self.device = device
        self.config = config
        self.damage_classes = {
            "un-classified": 0,  # 非建筑部分
            "no-damage": 1,
            "minor-damage": 2,
            "major-damage": 3,
            "destroyed": 4
        }

    def process_with_building_attention(self, prediction_prob, building_mask):
        """
        对小型建筑物进行特殊处理，当预测为unclassified时选择第二可能的类别
        Args:
            prediction_prob: softmax输出的概率分布 (H, W, 5)
            building_mask: 建筑物掩码
        """
        # 先获取argmax的预测结果
        prediction = np.argmax(prediction_prob, axis=2)
        
        if prediction.shape != building_mask.shape:
            raise ValueError("Prediction and building_mask must have the same shape")
                
        enhanced_pred = prediction.copy()
        # 2. 给每个独立建筑物打标签
        building_labels = measure.label(building_mask)
        
        print("\n处理小型建筑物...")
        for building_id in tqdm(range(1, building_labels.max() + 1), desc="分析小型建筑"):
            curr_building_mask = building_labels == building_id
            building_size = np.sum(curr_building_mask)
            
            if building_size < self.config['SMALL_BUILDING_THRESHOLD']:
                # 获取建筑物的边界框
                props = measure.regionprops(curr_building_mask.astype(int))
                bbox = props[0].bbox
                
                # 扩展感受野
                y1, x1, y2, x2 = bbox
                pad = self.config['CONTEXT_WINDOW'] // 2
                y1_ext = max(0, y1 - pad)
                x1_ext = max(0, x1 - pad)
                y2_ext = min(prediction.shape[0], y2 + pad)
                x2_ext = min(prediction.shape[1], x2 + pad)
                
                # 获取当前建筑物的预测和概率
                current_pred = prediction[curr_building_mask]
                current_probs = prediction_prob[curr_building_mask]
                
                # 只处理未分类的建筑物
                if np.all(current_pred == 0):
                    # 获取每个像素第二高概率的类别
                    sorted_indices = np.argsort(current_probs, axis=1)  # 按概率排序
                    second_best_classes = sorted_indices[:, -2]  # 获取第二高概率的类别
                    
                    if len(second_best_classes) > 0:
                        # 获取最常见的第二可能类别
                        main_second_class = np.bincount(second_best_classes).argmax()
                        if main_second_class > 0:  # 如果不是未分类
                            enhanced_pred[curr_building_mask] = main_second_class
                        
        return enhanced_pred

    def post_process(self, prediction_prob, building_mask, use_damage_threshold=False):
        """
        后处理函数，提高对损坏类别的检测敏感度
        Args:
            prediction_prob: softmax输出的概率分布 (H, W, 5)
            building_mask: 建筑物掩码
            use_damage_threshold: 是否使用damage threshold增强
        Returns:
            处理后的预测结果
        """
        # 先获取argmax的预测结果
        processed_pred = np.argmax(prediction_prob, axis=2)
        
        # 确保非建筑区域为0
        processed_pred[building_mask == 0] = self.damage_classes["un-classified"]
        #########################################################处理小型建筑的功能#################################################
        # # 先应用小型建筑物的特殊处理（只在building_mask内）
        # small_building_pred = self.process_with_building_attention(prediction_prob, building_mask)
        # # 只更新building_mask内的区域
        # processed_pred[building_mask > 0] = small_building_pred[building_mask > 0]
        #########################################################处理小型建筑的功能#################################################
        # 对所有建筑物（包括大型建筑）进行处理
        building_labels = measure.label(building_mask)
        print("\npost process...")
        for building_id in tqdm(range(1, building_labels.max() + 1), desc="deal with building"):
            building_mask = building_labels == building_id
            
            if use_damage_threshold:
                # 获取当前建筑物的概率分布
                building_probs = prediction_prob[building_mask]
                
                # # 对损伤类别进行加权（每个等级使用不同阈值）
                # weighted_probs = building_probs.copy()
                # minor damage
                weighted_probs[:, 2] *= self.config['DAMAGE_THRESHOLD']
                # major damage
                weighted_probs[:, 3] *= self.config['DAMAGE_THRESHOLD'] * 1.5
                # destroyed
                weighted_probs[:, 4] *= self.config['DAMAGE_THRESHOLD'] * 2.0
                
                # 对每个像素点分别取最大概率的类别
                pixel_pred = np.argmax(weighted_probs, axis=1)
                # pixel_pred[pixel_pred == 0] = self.damage_classes["un-classified"]
                damage_levels = pixel_pred[pixel_pred > 0]
                # 统计整个建筑物内最常见的类别
                if len(damage_levels) > 0:
                    main_damage = np.bincount(damage_levels).argmax()
                else:
                    main_damage = self.damage_classes["no-damage"]
            else:
                # 使用原来的处理方式
                building_pred = processed_pred[building_mask]
                damage_levels = building_pred[building_pred > 0]
                
                if len(damage_levels) > 0:
                    # 找出最常见的损坏等级
                    main_damage = np.bincount(damage_levels.astype(int)).argmax()
                else:
                    main_damage = self.damage_classes["no-damage"]
            
            # 将整个建筑物设置为该损坏等级
            processed_pred[building_mask] = main_damage
        
        # 验证输出
        assert processed_pred.min() >= 0 and processed_pred.max() <= 4, "Invalid prediction values"
        return processed_pred
        
   
def get_fixed_size_window(image, x1, y1, end_x, end_y, window_size=CONFIG['WINDOW_SIZE']):
    """
    获取固定尺寸的窗口
    Args:
        image: PIL Image
        x1, y1, end_x, end_y: 窗口坐标
        window_size: 目标窗口大小
    Returns:
        fixed_size_window: PIL Image with fixed size
    """
    # 创建固定尺寸的空白图像
    if image.mode == 'L':  # 对于mask图像
        fixed_window = Image.new('L', (window_size, window_size), 0)
    else:  # 对于RGB图像
        fixed_window = Image.new('RGB', (window_size, window_size), 0)
    
    # 裁剪原始窗口
    window = image.crop((x1, y1, end_x, end_y))
    # 粘贴到固定尺寸图像上
    fixed_window.paste(window, (0, 0))
    return fixed_window

def predict_with_sliding_window(model, pre_image, post_image, building_mask=None, device='cuda'):
    """使用滑动窗口进行预测"""
    model.to(device)
    model.eval()
    
    width, height = pre_image.size
    window_size = CONFIG['WINDOW_SIZE']
    overlap = CONFIG['OVERLAP']
    stride = window_size - overlap
    
    # 修改为5个通道来存储softmax输出
    output = np.zeros((height, width, 5), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    if building_mask is not None:
        building_mask = building_mask.resize((width, height), Image.NEAREST)
        building_mask_np = np.array(building_mask)
    
    predictor = BuildingAwarePredictor(model, device) if building_mask is not None else None
    
    y_steps = (height - overlap) // stride + (1 if height % stride != 0 else 0)
    x_steps = (width - overlap) // stride + (1 if width % stride != 0 else 0)
    total_steps = y_steps * x_steps
    
    with tqdm(total=total_steps, desc='Processing windows') as pbar:
        for y in range(0, height, stride):
            for x in range(0, width, stride):
                end_y = min(y + window_size, height)
                end_x = min(x + window_size, width)
                y1, x1 = max(0, y), max(0, x)
                
                pre_window = get_fixed_size_window(pre_image, x1, y1, end_x, end_y, window_size)
                post_window = get_fixed_size_window(post_image, x1, y1, end_x, end_y, window_size)
                
                if building_mask is not None:
                    mask_window = get_fixed_size_window(
                        Image.fromarray(building_mask_np), 
                        x1, y1, end_x, end_y, 
                        window_size
                    )
                else:
                    mask_window = None
                #pred 为 softmax
                # 获取概率分布预测
                pred = predict_image(model, pre_window, post_window, mask_window, device)
                
                actual_height = end_y - y1
                actual_width = end_x - x1
                # 调整预测结果的大小
                pred = pred[:actual_height, :actual_width, :]
                
                output[y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1
                
                pbar.update(1)
                
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    # 计算平均概率
    valid_mask = counts > 0
    for i in range(5):  # 对每个类别进行平均
        output[:, :, i][valid_mask] /= counts[valid_mask]
    
    if building_mask is not None:
        # 应用建筑物感知的增强
        # enhanced_pred = predictor.process_with_building_attention(
        #     pred, building_mask_np)
        output = predictor.post_process(output, building_mask_np)
    
    return output
def predict_image(model, pre_image, post_image, building_mask=None, device='cuda'):
    """对单个窗口进行预测"""
    model.to(device)
    model.eval()
    
    # 获取原始尺寸
    original_size = pre_image.size
    
    # 计算能被32整除的新尺寸（因为VGG19网络有5次下采样，2^5=32）
    new_width = ((original_size[0] + 31) // 32) * 32
    new_height = ((original_size[1] + 31) // 32) * 32
    
    # 调整图像尺寸
    pre_image = pre_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    post_image = post_image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    
    pre_tensor = preprocess_image(pre_image, is_mask=False)
    post_tensor = preprocess_image(post_image, is_mask=False)
    
    if building_mask is not None:
        building_mask = building_mask.resize((new_width, new_height), Image.Resampling.NEAREST)
        building_mask = preprocess_image(building_mask, is_mask=True)
    
    pre_tensor = pre_tensor.unsqueeze(0).to(device)
    post_tensor = post_tensor.unsqueeze(0).to(device)
    if building_mask is not None:
        building_mask = building_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(pre_tensor, post_tensor)
        # 转换为概率分布
        pred_prob = F.softmax(output, dim=1)
        # 转换为numpy数组，调整维度顺序 from (C, H, W) to (H, W, C)
        pred_prob = pred_prob.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # 调整回原始尺寸
        if pred_prob.shape[:2] != (original_size[1], original_size[0]):
            pred_prob_resized = np.zeros((original_size[1], original_size[0], pred_prob.shape[2]))
            for i in range(pred_prob.shape[2]):
                pred_prob_resized[:,:,i] = np.array(
                    Image.fromarray(pred_prob[:,:,i]).resize(
                        (original_size[0], original_size[1]), 
                        Image.Resampling.BILINEAR
                    )
                )
            pred_prob = pred_prob_resized
        
        return pred_prob

def preprocess_image(image, is_mask=False):
    """
    预处理图像，保持固定尺寸
    """
    if not is_mask and image.mode != 'RGB':
        image = image.convert('RGB')
    
    if is_mask:
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    return transform(image)
def visualize_prediction(image, mask, prediction, light_background=True):
    """
    内存优化的可视化函数，保持原始分辨率
    Args:
        image: 原始图像
        mask: ground truth mask
        prediction: 预测结果
        light_background: 是否使用浅色背景 (#f8f8f8)
    """
    # 打印类别统计
    unique, counts = np.unique(prediction, return_counts=True)
    total_pixels = prediction.size
    print("\nClass distribution:")
    class_names = ['Unclassified', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
    for value, count in zip(unique, counts):
        percentage = (count / total_pixels) * 100
        print(f"{class_names[value]}: {count} pixels ({percentage:.2f}%)")

    # 定义颜色映射
    colors = ['black', 'blue', 'green', 'yellow', 'red']
    n_classes = 5
    cmap = mcolors.ListedColormap(colors[:n_classes])
    bounds = np.arange(n_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建更大的图形，并调整子图之间的间距
    fig = plt.figure(figsize=(24, 8))    
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.1)

    # 设置显示参数
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.resample'] = False

    # 创建子图
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    if light_background:
        for ax in [ax1, ax2, ax3]:
            ax.set_facecolor('#C0C0C0')
        # 修改cmap中的黑色为灰色
        colors = ['#C0C0C0', 'blue', 'green', 'yellow', 'red']
        cmap = mcolors.ListedColormap(colors[:n_classes])
    
    # 显示原始图像
    ax1.imshow(image)
    ax1.set_title('Original Image', pad=20)  # 增加标题和图像之间的间距
    ax1.axis('off')

    # 显示ground truth（如果有）
    if mask is not None:
        ax2.imshow(mask, cmap=cmap, norm=norm)
        ax2.set_title('Ground Truth', pad=20)
    else:
        ax2.set_visible(False)
    ax2.axis('off')

    # # 显示预测结果
    if not light_background:
        # 创建RGB掩码（使用uint8以节省内存）
        overlay = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        
        # 逐类别处理，避免一次性创建大数组
        for class_idx, color in enumerate(colors):
            if class_idx > 0:  # 跳过未分类
                mask = prediction == class_idx
                rgb_color = np.array(mcolors.to_rgb(color)) * 255
                for i in range(3):
                    overlay[mask, i] = int(rgb_color[i])
        
        # 显示原始图像和叠加结果
        ax3.imshow(image)
        ax3.imshow(overlay, alpha=0.5)
        dummy_im = ax3.imshow(prediction, cmap=cmap, norm=norm, visible=False)
        cbar = fig.colorbar(dummy_im, ax=ax3, orientation='horizontal', 
                          fraction=0.046, pad=0.08)
    else:
        # 直接显示预测结果，不显示底图
        im3 = ax3.imshow(prediction, cmap=cmap, norm=norm)
        cbar = fig.colorbar(im3, ax=ax3, orientation='horizontal', 
                          fraction=0.046, pad=0.08)

    ax3.set_title('Prediction', pad=20)
    ax3.axis('off')
    cbar.set_ticks(bounds[:-1] + 0.5)
    cbar.set_ticklabels(class_names)
    cbar.ax.tick_params(labelsize=8)

    # 设置所有子图的纵横比相同并确保填充整个可用空间
    for ax in [ax1, ax2, ax3]:
        if ax.get_visible():
            ax.set_aspect('equal', adjustable='box')
            # 移除所有边距
            ax.set_position(ax.get_position().expanded(1.0, 1.0))

    # 移除 tight_layout，改用手动调整
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.02, right=0.98)

    # 确保像素级显示
    def on_draw(event):
        for ax in [ax1, ax2, ax3]:
            if ax.get_visible():
                ax.images[0].set_interpolation('nearest')

    fig.canvas.mpl_connect('draw_event', on_draw)
    plt.show()

def calculate_metrics(prediction, ground_truth):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 计算5分类指标
    prediction = prediction.astype(np.int64)
    ground_truth = ground_truth.astype(np.int64)
    prediction_tensor = torch.from_numpy(prediction).long().unsqueeze(0).to(device)
    ground_truth_tensor = torch.from_numpy(ground_truth).long().unsqueeze(0).to(device)
    
    # 修改混淆矩阵计算，只考虑建筑区域的类别
    confusion_matrix_5 = torchmetrics.ConfusionMatrix(
        task='multiclass', 
        num_classes=5,
        normalize='true'
    ).to(device)
    conf_mat_5 = confusion_matrix_5(prediction_tensor, ground_truth_tensor)
    # 只保留建筑物类别（移除背景类）
    conf_mat_5 = conf_mat_5[1:, 1:]
    
    # 初始化metrics（排除背景类）
    accuracy_5 = torchmetrics.Accuracy(task='multiclass', num_classes=5, 
                                     ignore_index=0, validate_args=False).to(device)
    precision_5 = torchmetrics.Precision(task='multiclass', num_classes=5, 
                                       ignore_index=0, average='macro', validate_args=False).to(device)
    recall_5 = torchmetrics.Recall(task='multiclass', num_classes=5, 
                                  ignore_index=0, average='macro', validate_args=False).to(device)
    f1_score_5 = torchmetrics.F1Score(task='multiclass', num_classes=5, 
                                     ignore_index=0, average='macro').to(device)
    f1_score_5_per_class = torchmetrics.F1Score(task='multiclass', num_classes=5, 
                                               ignore_index=0, average=None).to(device)
    iou_5 = JaccardIndex(task="multiclass", num_classes=5, 
                        ignore_index=0).to(device)
    
    # 更新metrics
    accuracy_5.update(prediction_tensor, ground_truth_tensor)
    precision_5.update(prediction_tensor, ground_truth_tensor)
    recall_5.update(prediction_tensor, ground_truth_tensor)
    f1_score_5.update(prediction_tensor, ground_truth_tensor)
    f1_score_5_per_class.update(prediction_tensor, ground_truth_tensor)
    iou_5.update(prediction_tensor, ground_truth_tensor)
    
    # 计算结果
    acc_5 = accuracy_5.compute()
    prec_5 = precision_5.compute()
    rec_5 = recall_5.compute()
    f1_5 = f1_score_5.compute()
    f1_5_per_class = f1_score_5_per_class.compute()
    iou_score_5 = iou_5.compute()
    
    # 创建二分类版本 (保持0为背景)
    binary_prediction = prediction.copy()
    binary_ground_truth = ground_truth.copy()
    
    # 将2,3,4类转换为2（损坏类）
    binary_prediction[(binary_prediction == 2) | (binary_prediction == 3) | (binary_prediction == 4)] = 2
    binary_ground_truth[(binary_ground_truth == 2) | (binary_ground_truth == 3) | (binary_ground_truth == 4)] = 2
    
    # 转换为tensor
    binary_prediction_tensor = torch.from_numpy(binary_prediction).long().unsqueeze(0).to(device)
    binary_ground_truth_tensor = torch.from_numpy(binary_ground_truth).long().unsqueeze(0).to(device)
    
    # 初始化二分类metrics（排除背景类）
    accuracy_2 = torchmetrics.Accuracy(task='multiclass', num_classes=3, 
                                     ignore_index=0, validate_args=False).to(device)
    precision_2 = torchmetrics.Precision(task='multiclass', num_classes=3, 
                                       ignore_index=0, average='macro', validate_args=False).to(device)
    recall_2 = torchmetrics.Recall(task='multiclass', num_classes=3, 
                                  ignore_index=0, average='macro', validate_args=False).to(device)
    f1_score_2 = torchmetrics.F1Score(task='multiclass', num_classes=3, 
                                     ignore_index=0, average='macro').to(device)
    f1_score_2_per_class = torchmetrics.F1Score(task='multiclass', num_classes=3, 
                                               ignore_index=0, average=None).to(device)
    iou_2 = JaccardIndex(task="multiclass", num_classes=3, 
                        ignore_index=0).to(device)
    
    # 更新二分类metrics
    accuracy_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    precision_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    recall_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    f1_score_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    f1_score_2_per_class.update(binary_prediction_tensor, binary_ground_truth_tensor)
    iou_2.update(binary_prediction_tensor, binary_ground_truth_tensor)
    
    # 计算二分类结果
    acc_2 = accuracy_2.compute()
    prec_2 = precision_2.compute()
    rec_2 = recall_2.compute()
    f1_2 = f1_score_2.compute()
    f1_2_per_class = f1_score_2_per_class.compute()
    iou_score_2 = iou_2.compute()
    
    # 二分类的混淆矩阵，同样移除背景类
    confusion_matrix_2 = torchmetrics.ConfusionMatrix(
        task='multiclass',
        num_classes=3,
        normalize='true'
    ).to(device)
    conf_mat_2 = confusion_matrix_2(binary_prediction_tensor, binary_ground_truth_tensor)
    # 只保留建筑物类别（移除背景类）
    conf_mat_2 = conf_mat_2[1:, 1:]
    
    # 清除缓存
    for metric in [accuracy_5, precision_5, recall_5, f1_score_5, iou_5, f1_score_5_per_class,
                  accuracy_2, precision_2, recall_2, f1_score_2, iou_2, f1_score_2_per_class]:
        metric.reset()
    
    return (acc_5.item(), f1_5.item(), iou_score_5.item(), prec_5.item(), rec_5.item(), f1_5_per_class.tolist(),
            acc_2.item(), f1_2.item(), iou_score_2.item(), prec_2.item(), rec_2.item(), f1_2_per_class.tolist(),
            conf_mat_5.cpu().numpy(), conf_mat_2.cpu().numpy())

def plot_confusion_matrices(conf_mat_5, conf_mat_2, save_path='./confusion_matrices.png'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # 修改类别名称，移除背景类
    class_names_5 = ['No Damage', 'Minor Dmg', 'Major Dmg', 'Destroyed']
    class_names_2 = ['No Damage', 'Damaged']
    
    # 创建一个图形，包含两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 5分类混淆矩阵
    sns.heatmap(conf_mat_5, annot=True, fmt='.3f', 
                xticklabels=class_names_5, 
                yticklabels=class_names_5,
                cmap='Blues', ax=ax1)
    ax1.set_title('4-Class Confusion Matrix (Building Area Only)')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2分类混淆矩阵
    sns.heatmap(conf_mat_2, annot=True, fmt='.3f', 
                xticklabels=class_names_2, 
                yticklabels=class_names_2,
                cmap='Blues', ax=ax2)
    ax2.set_title('Binary Confusion Matrix (Building Area Only)')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
def calculate_building_f1_by_size(prediction, ground_truth, building_mask):
    """计算不同大小建筑物的F1分数统计信息"""
    # 标记每个独立的建筑物
    building_labels = measure.label(building_mask)
    max_size = 100  # 限制最大建筑物大小为100像素
    
    # 创建存储结构
    building_data = []
    
    # 为每个建筑物计算F1分数
    for building_id in range(1, building_labels.max() + 1):
        curr_building_mask = building_labels == building_id
        building_size = np.sum(curr_building_mask)
        
        if building_size > max_size:
            continue
            
        # 获取当前建筑物的预测和真实值
        curr_pred = prediction[curr_building_mask]
        curr_truth = ground_truth[curr_building_mask]
        
        # 转换为PyTorch张量计算F1分数
        pred_tensor = torch.from_numpy(curr_pred).long()
        truth_tensor = torch.from_numpy(curr_truth).long()
        
        # 计算F1分数
        f1_metric = F1Score(task='multiclass', num_classes=5, average='macro')
        f1_score = f1_metric(pred_tensor.unsqueeze(0), truth_tensor.unsqueeze(0)).item()
        
        building_data.append({
            'size': building_size,
            'f1_score': f1_score
        })
    
    # 转换为数组形式
    sizes = np.array([b['size'] for b in building_data])
    f1_scores = np.array([b['f1_score'] for b in building_data])
    
    # 创建大小bins (0-20, 21-40, 41-60, 61-80, 81-100)
    size_bins = np.arange(0, 101, 20)
    bin_labels = [f'{size_bins[i]}-{size_bins[i+1]-1}' for i in range(len(size_bins)-1)]
    
    # 计算每个bin的统计信息
    stats = []
    for i in range(len(size_bins)-1):
        mask = (sizes >= size_bins[i]) & (sizes < size_bins[i+1])
        if np.any(mask):
            bin_f1_scores = f1_scores[mask]
            stats.append({
                'bin': bin_labels[i],
                'avg_f1': float(np.mean(bin_f1_scores)),  # 确保数据是Python原生类型
                'std_f1': float(np.std(bin_f1_scores)),
                'count': int(np.sum(mask))
            })
    
    print("\nBuilding size analysis:")
    for stat in stats:
        print(f"Size {stat['bin']}: Avg F1 = {stat['avg_f1']:.3f} ± {stat['std_f1']:.3f} (n={stat['count']})")
    
    return stats

def plot_building_size_analysis(stats, save_path):
    """创建并保存建筑物大小分析图表"""
    plt.figure(figsize=(12, 6))
    
    # 准备数据
    bins = [s['bin'] for s in stats]
    counts = [s['count'] for s in stats]
    avg_f1s = [s['avg_f1'] for s in stats]
    std_f1s = [s['std_f1'] for s in stats]
    
    # 创建双Y轴图表
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    # 绘制柱状图（建筑物数量）
    bars = ax1.bar(bins, counts, alpha=0.6, color='#8884d8')
    ax1.set_xlabel('Building Size (pixels)')
    ax1.set_ylabel('Building Count', color='#8884d8')
    ax1.tick_params(axis='y', labelcolor='#8884d8')
    
    # 绘制折线图（F1分数）
    line = ax2.errorbar(bins, avg_f1s, yerr=std_f1s, color='#82ca9d', 
                       marker='o', linestyle='-', linewidth=2, 
                       capsize=5, capthick=2)
    ax2.set_ylabel('F1 Score', color='#82ca9d')
    ax2.tick_params(axis='y', labelcolor='#82ca9d')
    ax2.set_ylim(0, 1)
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        bars.patches[0],
        Line2D([0], [0], color='#82ca9d', marker='o', linestyle='-', 
               linewidth=2, label='Average F1 Score')
    ]
    ax1.legend(legend_elements, ['Building Count', 'Average F1 Score'])
    
    # 添加标题和网格
    plt.title('Building Size vs F1 Score Analysis')
    ax1.grid(True, alpha=0.3)
    
    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Building size analysis has been saved to {save_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-original', action='store_true',
                      help='Show original image for unclassified areas')
    parser.add_argument('--building-mask', type=str, default='',
                      help='Path to the building mask file (binary mask for detection)')
    parser.add_argument('--ground-truth-mask', type=str, default='',
                      help='Path to the ground truth mask file (for evaluation)')
    parser.add_argument('--light-background', action='store_true',
                      help='Use light background color')
    args = parser.parse_args()
    # Second BEST MODEL version 9.4 change loss to only Dice loss
    # model_path = './checkpoints/saved_94_enhanced_v_1.0_lr_5.0e-05_20250125_163021/best_model.pth'
    # version 8.4 change loss to only Floss loss
    # model_path = './checkpoints/saved_84_enhanced_v_1.0_lr_5.0e-05_20250120_204602/best_model.pth'
    # version 7.4 change loss to only Dice loss
    # model_path = './checkpoints/saved_74_enhanced_v_1.0_lr_5.0e-05_20250120_164924/best_model.pth'
    # version 6.4 change loss to CEL 
    # model_path = './checkpoints/saved_64_enhanced_v_1.0_lr_5.0e-05_20250114_204116/best_model.pth'
    # version 5.4 change loss to CEL + focal loss
    # model_path = './checkpoints/saved_54_enhanced_v_1.0_lr_5.0e-05_20250111_230010/best_model.pth'
    # version 4.4 change lr to 1e-6 from 1.x version
    # model_path = './checkpoints/enhanced_v_1.0_lr_1.0e-06_20241226_210303/checkpoint_epoch24.pth'
    # version 3.4 ONLY ON BUILDING AREA
    # model_path = './checkpoints/saved_34_enhanced_v_1.0_lr_5.0e-05_20241226_165039/best_model.pth'
    # version 2 loss only on building area
    # model_path = './checkpoints/saved_24_enhanced_v_1.0_lr_5.0e-05_20241224_212710/checkpoint_epoch25.pth'

    ####### compare for generate RESULT OUTPUT
    # VGG19
    # pretrain + fine-tuning BEST MODEL !!!version 1.4 loss: Dice+CEL
    model_path = './checkpoints/saved_14_enhanced_v_1.0_lr_5.0e-05_20241221_171929/best_model.pth'
    # ONLY pretrain version 0 
    # model_path = './training/checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch60.pth'
    # baseline
    # model_path = ''
    # ONLY FINE-TUNING is training
    # model_path = ''

    #好像下面这个RESNET的
    # model_path = './training/checkpoints/best0921.pth'
    
    ### DATASET
    # xBD dataset
    # img_home_path = "C:/Users/xiao/peng/xbd/Dataset/Validation"
    # pre_image_path = img_home_path + "/Pre/Image512/"+ "hurricane-michael_00000400_pre_disaster.png"
    # post_image_path = img_home_path + "/Post/Image512/"+ "hurricane-michael_00000400_post_disaster.png"
    # mask_path = img_home_path + "/Post/Label512/"+ "hurricane-michael_00000400_post_disaster.png"

    # my dataset
    #   Volnovakha
    # pre_image_path = './training/images/Volnovakha_20210513.tif'
    pre_image_path = './training/images/Volnovakha_20210622_20220512_pre.tif'
    post_image_path = './training/images/Volnovakha_20210622_20220512_post.tif'

    #   Rubizhne
    # pre_image_path = './training/images/Pre/Image512/Rubizhne_20210915_20220921_pre.tif'
    # post_image_path = './training/images/Post/Image512/Rubizhne_20210915_20220921_post.tif'
    #pre
    #./images/20210709_073742_79_2431_3B_Visual_clip.tif
    #./images/75cm_Bilinear_20210709.tif
    # pre_image_path = './training/images/20210915_073716_84_2440_3B_Visual_clip.tif'
    #post
    #./images/20220713_075021_72_222f_3B_Visual_clip.tif
    #./images/20220709_072527_82_242b_3B_Visual_clip.tif
    # 75cm
    #./images/75cm_Bilinear.tif
    #./images/75cm_Bilinear_20220921.tif

    # #my dataset2
    # pre_image_path = './images/before_Zhovteneyvi.jpeg'
    # post_image_path = './images/after_Zhovteneyvi.jpeg'
    # mask_path = ''

    # mask_path = 'path/to/ground_truth_mask.tif'  # 如果有的话

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path)
    model.to(device)
    
    pre_image = Image.open(pre_image_path)
    post_image = Image.open(post_image_path)
    building_mask = None
    if args.building_mask:
        try:
            building_mask = Image.open(args.building_mask)
            print("Successfully loaded building mask")
        except Exception as e:
            print(f"Error loading building mask: {e}")
            building_mask = None
    
    ground_truth_mask = None
    if args.ground_truth_mask:
        try:
            ground_truth_mask = Image.open(args.ground_truth_mask)
            ground_truth_np = np.array(ground_truth_mask)
            # 确保mask是单通道的
            if len(ground_truth_np.shape) > 2:
                ground_truth_np = ground_truth_np[:,:,0]  # 取第一个通道
            print("Successfully loaded ground truth mask")
        except Exception as e:
            print(f"Error loading ground truth mask: {e}")
            ground_truth_mask = None
    
    # 使用CONFIG中的WINDOW_SIZE参数
    if pre_image.size[0] > CONFIG['WINDOW_SIZE'] or pre_image.size[1] > CONFIG['WINDOW_SIZE']:
        prediction = predict_with_sliding_window(
            model, pre_image, post_image, building_mask,
            device=device
        )
    else:
        # without sliding window and any post-processing
        pred_prob = predict_image(model, pre_image, post_image, 
                                building_mask, device)
        # 将概率转换为类别标签
        prediction = np.argmax(pred_prob, axis=2)
        
        # 如果有建筑物掩码，将非建筑区域设为0
        if building_mask is not None:
            building_mask_np = np.array(building_mask)
            prediction[building_mask_np == 0] = 0
    
    # 评估
    if ground_truth_mask is not None:
        # 在进行预测之前调整ground truth的尺寸
        ground_truth_mask = ground_truth_mask.resize(
            (post_image.size[0], post_image.size[1]), 
            Image.NEAREST
        )
        ground_truth_np = np.array(ground_truth_mask)
        metrics = calculate_metrics(prediction, ground_truth_np)
        if building_mask is not None:
            print("\nAnalyzing building size distribution...")
            stats = calculate_building_f1_by_size(prediction, ground_truth_np, 
                                                np.array(building_mask))
            
            # 保存建筑物大小分析图表
            size_analysis_path = f'./building_size_analysis_{Path(pre_image_path).stem}.png'
            plot_building_size_analysis(stats, size_analysis_path)
            print(f"Building size analysis has been saved to {size_analysis_path}")        
        print("\nMulti-classes evaluation result:")
        print(f'Accuracy: {metrics[0]:.4f}')
        print(f'F1 Score: {metrics[1]:.4f}')
        print(f'IoU: {metrics[2]:.4f}')
        print(f'Precision: {metrics[3]:.4f}')
        print(f'Recall: {metrics[4]:.4f}')
        print("\nEach class f1 score:")
        class_names_5 = ['Background', 'No Damage', 'Minor Damage', 'Major Damage', 'Destroyed']
        for i, f1 in enumerate(metrics[5]):
            print(f'{class_names_5[i]}: {f1:.4f}')
        print("\n binary-classes (undamaged vs damaged):")
        print(f'Accuracy: {metrics[6]:.4f}')
        print(f'F1 Score: {metrics[7]:.4f}')
        print(f'IoU: {metrics[8]:.4f}')
        print(f'Precision: {metrics[9]:.4f}')
        print(f'Recall: {metrics[10]:.4f}')
        print("\nEach class f1 score:")
        class_names_2 = ['Background', 'Undamaged', 'Damaged']
        for i, f1 in enumerate(metrics[11]):
            print(f'{class_names_2[i]}: {f1:.4f}')
        visualize_prediction(np.array(post_image), ground_truth_np, prediction, args.light_background)

        # 保存混淆矩阵热力图
        # 可以根据不同的模型或数据集生成不同的文件名
        save_path = f'./confusion_matrices_{Path(pre_image_path).stem}.png'
        plot_confusion_matrices(metrics[-2], metrics[-1], save_path)
        print(f"Confusion matrices have been saved to {save_path}")

    else:
        visualize_prediction(np.array(post_image), None, prediction, args.light_background)

if __name__ == '__main__':
    main()