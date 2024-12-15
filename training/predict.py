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
# 在文件开头添加必要的导入
from skimage import measure
import torch.nn.functional as F
from scipy import ndimage

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
CONFIG = {
    'SMALL_BUILDING_THRESHOLD': 100,   # Pixel threshold for small buildings at 3m resolution
    'WINDOW_SIZE': 512,    # Sliding window size
    'OVERLAP': 32,         # Overlap region size
    'CONTEXT_WINDOW': 64,  # Context window size
    'DAMAGE_THRESHOLD': 0.3,# Damage assessment threshold
}
def load_model(model_path):
    #model = SiameseUNetWithResnet50Encoder()
    model = SiamUNetConCVgg19()
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

    def post_process(self, prediction_prob, building_mask):
        """
        后处理函数，提高对损坏类别的检测敏感度
        Args:
            prediction_prob: softmax输出的概率分布 (H, W, 5)
            building_mask: 建筑物掩码
        Returns:
            处理后的预测结果
        """
        # 先获取argmax的预测结果
        processed_pred = np.argmax(prediction_prob, axis=2)
        
        # 确保非建筑区域为0
        processed_pred[building_mask == 0] = self.damage_classes["un-classified"]
        
        # 先应用小型建筑物的特殊处理
          # 先应用小型建筑物的特殊处理（只在building_mask内）
        small_building_pred = self.process_with_building_attention(prediction_prob, building_mask)
        # 只更新building_mask内的区域
        processed_pred[building_mask > 0] = small_building_pred[building_mask > 0]
        # 对所有建筑物（包括大型建筑）进行处理
        building_labels = measure.label(building_mask)
        print("\n开始后处理优化...")
        for building_id in tqdm(range(1, building_labels.max() + 1), desc="处理建筑物"):
            building_mask = building_labels == building_id
            building_pred = processed_pred[building_mask]
            damage_levels = building_pred[building_pred > 0]
            
            if len(damage_levels) > 0:
                main_damage = np.bincount(damage_levels.astype(int)).argmax()
                processed_pred[building_mask] = main_damage
            else:
                processed_pred[building_mask] = self.damage_classes["no-damage"]
        
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
    
    pre_tensor = preprocess_image(pre_image, is_mask=False)
    post_tensor = preprocess_image(post_image, is_mask=False)
    
    if building_mask is not None:
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
def visualize_prediction(image, mask, prediction, show_original_unclassified=False):
    """
    内存优化的可视化函数，保持原始分辨率
    Args:
        image: 原始图像数组
        mask: Ground truth掩码数组（或None）
        prediction: 预测数组
        show_original_unclassified: 是否在未分类区域显示原始图像
    """
    # 打印类别统计
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
    bounds = np.arange(n_classes + 1)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 创建图形
    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.3)

    # 设置显示参数
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.resample'] = False

    # 创建子图
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # 显示原始图像
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax1.axis('off')

    # 显示ground truth（如果有）
    if mask is not None:
        ax2.imshow(mask, cmap=cmap, norm=norm)
        ax2.set_title('Ground Truth')
    else:
        ax2.set_visible(False)
    ax2.axis('off')

    # 显示预测结果
    if show_original_unclassified:
        # 创建RGB掩码（使用uint8以节省内存）
        overlay = np.zeros((*prediction.shape, 3), dtype=np.uint8)
        
        # 逐类别处理，避免一次性创建大数组
        for class_idx, color in enumerate(colors):
            if class_idx > 0:  # 跳过未分类
                mask = prediction == class_idx
                rgb_color = np.array(mcolors.to_rgb(color)) * 255
                for i in range(3):
                    overlay[mask, i] = int(rgb_color[i])
        
        # 显示原始图像
        ax3.imshow(image)
        # 叠加预测结果
        ax3.imshow(overlay, alpha=0.5)
    else:
        # 直接显示预测结果
        im3 = ax3.imshow(prediction, cmap=cmap, norm=norm)

    ax3.set_title('Prediction')
    ax3.axis('off')

    # 添加颜色条
    if show_original_unclassified:
        dummy_im = ax3.imshow(prediction, cmap=cmap, norm=norm, visible=False)
        cbar = fig.colorbar(dummy_im, ax=ax3, orientation='horizontal', 
                          fraction=0.046, pad=0.04)
    else:
        cbar = fig.colorbar(im3, ax=ax3, orientation='horizontal', 
                          fraction=0.046, pad=0.04)

    cbar.set_ticks(bounds[:-1] + 0.5)
    cbar.set_ticklabels(class_names)

    # 设置纵横比
    for ax in [ax2, ax3]:
        if ax.get_visible():
            ax.set_aspect('equal', adjustable='box')

    # 确保像素级显示
    def on_draw(event):
        for ax in [ax2, ax3]:
            if ax.get_visible():
                ax.images[0].set_interpolation('nearest')

    fig.canvas.mpl_connect('draw_event', on_draw)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-original', action='store_true',
                      help='Show original image for unclassified areas')
    parser.add_argument('--building-mask', type=str, default='',
                      help='Path to the building mask file (binary mask for detection)')
    parser.add_argument('--ground-truth-mask', type=str, default='',
                      help='Path to the ground truth mask file (for evaluation)')
    args = parser.parse_args()
    model_path = './training/checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch60.pth'
    #model_path = './training/checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch52.pth'
    #好像下面这个RESNET的
    #model_path = './training/checkpoints/best0921.pth'
    # # xbd
    # img_home_path = "C:/Users/xiao/peng/xbd/Dataset/Validation"
    # pre_image_path = img_home_path + "/Pre/Image512/"+ "hurricane-michael_00000400_pre_disaster.png"
    # post_image_path = img_home_path + "/Post/Image512/"+ "hurricane-michael_00000400_post_disaster.png"
    # mask_path = img_home_path + "/Post/Label512/"+ "hurricane-michael_00000400_post_disaster.png"

    #my dataset
    #pre
    #./images/20210709_073742_79_2431_3B_Visual_clip.tif
    #./images/75cm_Bilinear_20210709.tif
    #post
    #./images/20220713_075021_72_222f_3B_Visual_clip.tif
    #./images/20220709_072527_82_242b_3B_Visual_clip.tif

    #./images/75cm_Bilinear.tif
    #./images/75cm_Bilinear_20220921.tif
    # pre_image_path = './training/images/20210915_073716_84_2440_3B_Visual_clip.tif'
    pre_image_path = './training/images/20210915_073716_84_2440_3B_Visual_clip.tif'
    post_image_path = './training/images/20220921_080914_68_2254_3B_Visual_clip.tif'

    mask_path = ''

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
        prediction = predict_image(model, pre_image, post_image, 
                                 building_mask, device)
    
    # 评估
    if ground_truth_mask is not None:
        ground_truth_np = np.array(ground_truth_mask)
        visualize_prediction(np.array(post_image), ground_truth_np, prediction, args.show_original)
        dice, f1, iou, precision, recall = calculate_metrics(prediction, ground_truth_np)
        print(f'Dice Score: {dice:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'IoU: {iou:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
    else:
        visualize_prediction(np.array(post_image), None, prediction, args.show_original)

if __name__ == '__main__':
    main()