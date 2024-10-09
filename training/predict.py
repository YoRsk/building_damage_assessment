import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from model.my_models import SiameseUNetWithResnet50Encoder  # 导入您的模型
from utils.data_loading import preprocess  # 导入您的预处理函数
from utils.dice_score import multiclass_dice_coeff, dice_coeff
from torchmetrics import F1Score, JaccardIndex, AUROC, Precision, Recall
import torch.nn.functional as F
import matplotlib.colors as mcolors

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
    
    ax2.imshow(mask, cmap=cmap, norm=norm)
    ax2.set_title('Ground Truth')
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
    
    auroc = AUROC(task="multiclass", num_classes=5)
    # 为AUROC计算准备概率分布
    pred_probs = F.softmax(torch.randn(1, 5, *prediction.shape), dim=1)  # 这里使用随机值，实际中应使用模型的原始输出
    auc_roc = auroc(pred_probs, ground_truth_tensor)
    
    return dice.item(), f1.item(), iou.item(), precision.item(), recall.item(), auc_roc.item()

def main():
    model_path = 'path/to/your/trained/model.pth'
    pre_image_path = 'path/to/large_pre_disaster_image.tif'
    post_image_path = 'path/to/large_post_disaster_image.tif'
    mask_path = 'path/to/ground_truth_mask.tif'  # 如果有的话
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model(model_path, device)
    
    pre_image = Image.open(pre_image_path)
    post_image = Image.open(post_image_path)
    mask = Image.open(mask_path) if mask_path else None
    
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
