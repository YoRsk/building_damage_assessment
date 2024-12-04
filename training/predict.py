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

# 0.75m分辨率推荐配置
# CONFIG = {
#     'SMALL_BUILDING_THRESHOLD': 1500,  # 根据0.75m分辨率调整，约等于面积843.75平方米
#     'WINDOW_SIZE': 512,     # 滑动窗口大小，覆盖实际地面约768米
#     'OVERLAP': 128,         # 增加重叠以减少边界效应，实际约96米重叠
#     'CONTEXT_WINDOW': 256,  # 上下文窗口大小，考虑周围约96米范围
#     'DAMAGE_THRESHOLD': 0.3, # 损伤判断阈值
# }

#3m分辨率推荐配置
CONFIG = {
    'SMALL_BUILDING_THRESHOLD': 15,   # 根据3m分辨率调整，约等于面积900平方米
    'WINDOW_SIZE': 512,    # 滑动窗口大小，覆盖实际地面约3072米
    'OVERLAP': 32,         # 重叠区域大小，实际约96米重叠
    'CONTEXT_WINDOW': 64,  # 上下文窗口，考虑周围约192米范围
    'DAMAGE_THRESHOLD': 0.3,# 损伤判断阈值
}
def load_model(model_path):
    # model = SiameseUNetWithResnet50Encoder()
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

    def process_with_building_attention(self, prediction, building_mask):
        """对小型建筑物进行特殊处理，当预测为unclassified时选择第二可能的类别"""
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
                
                # 分析周围区域，包括当前建筑物
                context_region = prediction[y1_ext:y2_ext, x1_ext:x2_ext]
                # 先看当前建筑物的预测情况
                current_pred = prediction[curr_building_mask]
                
                # 只处理未分类的建筑物
                if np.all(current_pred == 0):
                    unique_classes, class_counts = np.unique(context_region, return_counts=True)
                    # 按频次降序排序
                    sort_idx = np.argsort(class_counts)[::-1]
                    sorted_classes = unique_classes[sort_idx]
                    
                    # 如果第一个类别是未分类(0)且存在第二个类别，使用第二个类别
                    if sorted_classes[0] == 0 and len(sorted_classes) > 1:
                        enhanced_pred[curr_building_mask] = max(1, sorted_classes[1])
                    else:
                        # 如果只有未分类，设为no-damage
                        enhanced_pred[curr_building_mask] = self.damage_classes["no-damage"]
                        
        return enhanced_pred
        # """对小型建筑物进行特殊处理"""
        # enhanced_pred = prediction.copy()
        # building_labels = measure.label(building_mask)
        
        # for building_id in range(1, building_labels.max() + 1):
        #     building_mask = building_labels == building_id
        #     building_size = np.sum(building_mask)
            
        #     if building_size < self.config['SMALL_BUILDING_THRESHOLD']:
        #         # 获取建筑物的边界框
        #         props = measure.regionprops(building_mask.astype(int))
        #         bbox = props[0].bbox
                
        #         # 扩展感受野
        #         y1, x1, y2, x2 = bbox
        #         pad = self.config['CONTEXT_WINDOW'] // 2
        #         y1_ext = max(0, y1 - pad)
        #         x1_ext = max(0, x1 - pad)
        #         y2_ext = min(prediction.shape[0], y2 + pad)
        #         x2_ext = min(prediction.shape[1], x2 + pad)
                
        #         # 分析周围建筑物的损伤情况
        #         context_region = prediction[y1_ext:y2_ext, x1_ext:x2_ext]
        #         # 只考虑建筑物区域的损伤等级（非0的部分）
        #         context_damage = context_region[context_region > 0]
                
        #         if len(context_damage) > 0:
        #             # 计算损伤比例
        #             damage_ratio = len(context_damage) / context_region.size
                    
        #             if damage_ratio > self.config['DAMAGE_THRESHOLD']:
        #                 # 获取周围建筑物的主要损伤等级
        #                 damage_levels, counts = np.unique(context_damage, return_counts=True)
        #                 main_damage = damage_levels[counts.argmax()]
                        
        #                 # 当前建筑物的预测
        #                 current_pred = prediction[building_mask]
        #                 if np.all(current_pred == 0):  # 如果当前预测为未分类
        #                     # 根据周围损伤情况设置损伤等级
        #                     enhanced_pred[building_mask] = max(1, int(main_damage))
                        
        # return enhanced_pred

    def post_process(self, prediction, building_mask):
        """后处理优化"""
        processed_pred = prediction.copy()
        processed_pred[building_mask == 0] = self.damage_classes["un-classified"]
        
        # 先应用小型建筑物的特殊处理
        processed_pred = self.process_with_building_attention(processed_pred, building_mask)
        
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
        
        assert processed_pred.min() >= 0 and processed_pred.max() <= 4, "Invalid prediction values"
        return processed_pred

    # def process_with_building_attention(self, prediction, building_mask, window_size=64):
    #     """
    #     对小型建筑物进行特殊处理，考虑周围建筑物的损伤情况来调整预测结果
        
    #     Args:
    #         prediction (np.ndarray): 模型的预测结果，取值范围[0-4]
    #             0: 未分类
    #             1: 无损伤
    #             2: 轻微损伤
    #             3: 重度损伤
    #             4: 摧毁
    #         building_mask (np.ndarray): 建筑物掩码，1表示建筑物区域，0表示非建筑物区域
    #         window_size (int): 查看周围区域的窗口大小
            
    #     Returns:
    #         np.ndarray: 增强后的预测结果
    #     """
    #     enhanced_pred = prediction.copy()
    #     building_labels = measure.label(building_mask)  # 为每个独立的建筑物标注唯一ID
        
    #     for building_id in range(1, building_labels.max() + 1):
    #         # 获取当前建筑物的掩码
    #         building_mask = building_labels == building_id
    #         building_size = np.sum(building_mask)
            
    #         # 只处理小型建筑物
    #         if building_size < self.small_building_threshold:
    #             # 获取建筑物的边界框坐标
    #             props = measure.regionprops(building_mask.astype(int))
    #             bbox = props[0].bbox  # (min_row, min_col, max_row, max_col)
                
    #             # 先检查当前建筑物的预测情况
    #             current_pred = prediction[building_mask]
    #             # 如果当前预测为未分类(0)或无损伤(1)
    #             if np.all((current_pred == 0) | (current_pred == 1)):
    #                 # 扩展感受野，查看建筑物周围的区域
    #                 y1, x1, y2, x2 = bbox
    #                 pad = window_size // 2  # 向外扩展的像素数
    #                 y1_ext = max(0, y1 - pad)
    #                 x1_ext = max(0, x1 - pad)
    #                 y2_ext = min(prediction.shape[0], y2 + pad)
    #                 x2_ext = min(prediction.shape[1], x2 + pad)
                    
    #                 # 获取扩展区域内的预测结果
    #                 context_region = prediction[y1_ext:y2_ext, x1_ext:x2_ext]
    #                 # 获取周围有损伤的建筑物预测（值大于1的部分）
    #                 damage_context = context_region[context_region > 1]
                    
    #                 if len(damage_context) > 0:
    #                     # 计算周围区域的损伤比例
    #                     damage_ratio = len(damage_context) / context_region.size
                        
    #                     if damage_ratio > 0.2:  # 如果周围有显著损伤
    #                         # 统计损伤等级并获取最常见的损伤等级
    #                         damage_levels, counts = np.unique(damage_context, return_counts=True)
    #                         main_damage = damage_levels[counts.argmax()]
                            
    #                         # 如果周围主要损伤等级大于当前预测，更新预测结果
    #                         if main_damage > np.max(current_pred):
    #                             enhanced_pred[building_mask] = int(main_damage)
        
    #     return enhanced_pred

   
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
    
    output = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    
    if building_mask is not None:
        building_mask = building_mask.resize((width, height), Image.NEAREST)
        building_mask_np = np.array(building_mask)
    
    predictor = BuildingAwarePredictor(model, device) if building_mask is not None else None
    
    # 计算总步数
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
                #softmax
                pred = predict_image(model, pre_window, post_window, mask_window, device)
                
                actual_height = end_y - y1
                actual_width = end_x - x1
                pred = pred[:actual_height, :actual_width]
                
                output[y1:end_y, x1:end_x] += pred
                counts[y1:end_y, x1:end_x] += 1
                
                pbar.update(1)
                
                if device == 'cuda':
                    torch.cuda.empty_cache()
    
    valid_mask = counts > 0
    output[valid_mask] = np.round(output[valid_mask] / counts[valid_mask])
    output = output.astype(np.uint8)
    
    if building_mask is not None:
        # 应用建筑物感知的增强
        # enhanced_pred = predictor.process_with_building_attention(
        #     pred, building_mask_np)
        output = predictor.post_process(output, building_mask_np)
    
    return output
def predict_image(model, pre_image, post_image, building_mask=None, device='cuda'):
    """
    对单个窗口进行预测
    """
    model.to(device)
    model.eval()
    
    # 预处理
    pre_tensor = preprocess_image(pre_image, is_mask=False)
    post_tensor = preprocess_image(post_image, is_mask=False)
    
    if building_mask is not None:
        building_mask = preprocess_image(building_mask, is_mask=True)
    
    # 添加batch维度并移到设备
    pre_tensor = pre_tensor.unsqueeze(0).to(device)
    post_tensor = post_tensor.unsqueeze(0).to(device)
    if building_mask is not None:
        building_mask = building_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # 获取预测结果
        output = model(pre_tensor, post_tensor)
        pred_prob = F.softmax(output, dim=1).squeeze().cpu().numpy()
        
        # 获取初始预测
        # initial_pred = pred_prob.argmax(dim=1).squeeze().cpu().numpy()
        
        
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
    
    # model_path = './checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch49.pth'
    model_path = './training/checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch52.pth'
    # model_path = './checkpoints/best0921.pth'
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
    pre_image_path = './training/images/20210709_073742_79_2431_3B_Visual_clip.tif'
    post_image_path = './training/images/20220713_075021_72_222f_3B_Visual_clip.tif'

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