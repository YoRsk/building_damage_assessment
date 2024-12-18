from model.my_models import SiameseUNetWithResnet50Encoder, SiamUNetConCVgg19
from model.evaluate import evaluate
from utils.data_loading_enhanced import EnhancedSatelliteDataset
from utils.dice_score import dice_loss

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.focal import FocalLoss
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch.optim as optim
import torchmetrics
import logging
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from utils.tensor_encoder import TensorEncoder
from clearml import Task
from torchmetrics import JaccardIndex
closs = nn.CrossEntropyLoss()

floss = FocalLoss(mode = 'multiclass',
                alpha = None,
                gamma = 2.0,
                ignore_index = None,
                reduction = "mean",
                normalized = False,
                reduced_threshold = None)
# 设置日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('process_enhanced.log')
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

img_home_path = "C:/Users/xiao/PycharmProjects/building_damage_assessment/training/images"
train_data = {
    'pre_img': f"{img_home_path}/Pre/Image512/",
    'pre_mask': f"{img_home_path}/Pre/Label512/",
    'post_img': f"{img_home_path}/Post/Image512/",
    'post_mask': f"{img_home_path}/Post/Label512/"
}

def train_net_enhanced(net,
                      device,
                      epochs: int = 30,
                      batch_size: int = 64,
                      learning_rate: float = 1e-5,
                      train_loader=None,
                      val_loader=None,
                      save_checkpoint: bool = True,
                      gradient_clipping: float = 1.0):
    """使用增强数据集的训练函数"""
    
    # 使用传入的data loader
    if train_loader is None or val_loader is None:
        raise ValueError("train_loader and val_loader must be provided")
    
    # 创建目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f'enhanced_v_1.0_lr_{learning_rate:.1e}_{timestamp}'
    dir_checkpoint = Path(f'checkpoints/{dir_name}/')
    save_dir = Path(f'training_logs/{dir_name}/')
    log_dir = Path(f'tensorboard_logs/{dir_name}/')
    
    for dir_path in [dir_checkpoint, save_dir, log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(log_dir))

    # 设置优化器和学习率调度器
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    grad_scaler = torch.amp.GradScaler(enabled=ampbool)
    criterion = nn.CrossEntropyLoss()

    # 初始化训练指标
    train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5, validate_args=False).to(device)
    train_precision = torchmetrics.Precision(task='multiclass', num_classes=5, average='macro', validate_args=False).to(device)
    train_recall = torchmetrics.Recall(task='multiclass', num_classes=5, average='macro', validate_args=False).to(device)
    train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=5, average='macro').to(device)
    train_iou = JaccardIndex(task="multiclass", num_classes=5).to(device)

    # 用于跟踪已保存的文件
    saved_checkpoints = []
    saved_confusion_matrices = []
    saved_log_files = []

    # 创建训练数据记��字典
    training_data = {
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "ampbool": ampbool,
            "traintype": traintype,
            "gradient_clipping": gradient_clipping
        },
        "training_history": []
    }

    for epoch in range(start_epoch, start_epoch + epochs):
        net.train()
        epoch_loss = 0
        epoch_steps = 0 
        nancount = 0
        
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                preimage, postimage, post_masks, pre_masks = batch['preimage'], batch['postimage'], batch['postmask'], batch['premask']

                preimage = preimage.to(device=device, dtype=torch.float32)
                postimage = postimage.to(device=device, dtype=torch.float32)
                post_masks = post_masks.to(device=device, dtype=torch.long)
                pre_masks = pre_masks.to(device=device, dtype=torch.long)
                
                with torch.amp.autocast('cuda', enabled=ampbool):
                    masks_pred = None
                    if traintype == 'both':
                        masks_pred = net(preimage, postimage)
                        loss = criterion(masks_pred, post_masks)
                        loss += floss(masks_pred, post_masks)
                    elif traintype == 'pre':
                        masks_pred = net(preimage)
                        loss = criterion(masks_pred, pre_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(pre_masks, 2).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                    elif traintype == 'post':
                        masks_pred = net(postimage)
                        loss = criterion(masks_pred, post_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float()[:, 1:, ...],
                            F.one_hot(post_masks, 5).permute(0, 3, 1, 2).float()[:, 1:, ...],
                            multiclass=True
                        )

                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # Update metrics
                train_accuracy.update(masks_pred, post_masks)
                train_precision.update(masks_pred, post_masks)
                train_recall.update(masks_pred, post_masks)
                train_f1.update(masks_pred, post_masks)
                train_iou.update(masks_pred.argmax(dim=1), post_masks)
                
                pbar.update(postimage.shape[0])
                epoch_steps += 1
                
                if torch.isnan(loss):
                    epoch_loss += 0
                    nancount += 1
                else:
                    epoch_loss += loss.item()
                
                pbar.set_postfix(**{
                    'loss (batch)': loss.item(),
                    'loss': epoch_loss / epoch_steps,
                    'accuracy': train_accuracy.compute().item(),
                    'precision': train_precision.compute().item(),
                    'recall': train_recall.compute().item(),
                    'f1_score': train_f1.compute().item(),
                    'iou': train_iou.compute().item()
                })

        # 计算训练指标
        train_acc = train_accuracy.compute()
        train_prec = train_precision.compute()
        train_rec = train_recall.compute()
        train_f1_score = train_f1.compute()
        train_iou_score = train_iou.compute()
        train_loss = epoch_loss / len(train_loader)
        
        # 验证
        val_score, val_class_scores, val_loss, val_f1_macro, val_f1_per_class, val_iou, val_confusion_matrix = evaluate(
            net, val_loader, device, ampbool, traintype
        )

        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        # 记录epoch数据
        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_acc.item(),
            "val_accuracy": val_score,
            "train_precision": train_prec.item(),
            "train_recall": train_rec.item(),
            "train_f1_score": train_f1_score.item(),
            "val_f1_score_macro": val_f1_macro.item(),
            "val_f1_score_per_class": [f1.item() for f1 in val_f1_per_class],
            "val_class_scores": [score.item() for score in val_class_scores],
            "train_iou": train_iou_score.item(),
            "val_iou": val_iou.item(),
            "learning_rate": current_lr
        }
        training_data["training_history"].append(epoch_data)

        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_score, epoch)
        writer.add_scalar('Precision/train', train_prec, epoch)
        writer.add_scalar('Recall/train', train_rec, epoch)
        writer.add_scalar('F1 Score/train', train_f1_score, epoch)
        writer.add_scalar('F1 Score/validation_macro', val_f1_macro, epoch)
        writer.add_scalar('IoU/train', train_iou_score, epoch)
        writer.add_scalar('IoU/validation', val_iou, epoch)
        
        for i, f1 in enumerate(val_f1_per_class):
            writer.add_scalar(f'F1 Score/validation_class_{i}', f1, epoch)
        for i, class_score in enumerate(val_class_scores):
            writer.add_scalar(f'Dice Score/class_{i}', class_score, epoch)

        # 打印训练和验证结果
        print(f'\nEpoch {epoch}')
        print(f'Training - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}')
        print(f'F1 Score: {train_f1_score:.4f}, IoU: {train_iou_score:.4f}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation - Dice Score: {val_score:.4f}, F1 Score (macro): {val_f1_macro:.4f}, IoU: {val_iou:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'NaN Count: {nancount}')
        
        # 保存checkpoint
        if save_checkpoint:
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch{epoch}.pth'
            torch.save(net.state_dict(), str(checkpoint_path))
            saved_checkpoints.append(str(checkpoint_path))
            logger.info(f'Checkpoint {epoch} saved!')
            
            # 删除旧的检查点
            if len(saved_checkpoints) > 20:
                old_checkpoint = Path(saved_checkpoints.pop(0))
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f'Deleted old checkpoint: {old_checkpoint}')

        # 保存混淆矩阵
        # if val_confusion_matrix is not None:
        #     cm_path = save_dir / f"epoch_{epoch}_validation_confusion_matrix"
        #     save_confusion_matrix(val_confusion_matrix, str(save_dir), f"epoch_{epoch}_validation")
        #     saved_confusion_matrices.append(str(cm_path))
            
        #     if len(saved_confusion_matrices) > 20:
        #         old_cm = Path(saved_confusion_matrices.pop(0))
        #         if (old_cm.with_suffix('.png')).exists():
        #             (old_cm.with_suffix('.png')).unlink()
        #         if (old_cm.with_suffix('.npy')).exists():
        #             (old_cm.with_suffix('.npy')).unlink()

        # 定期保存training_data到log文件
        if epoch % 5 == 0 or epoch == epochs - 1:
            filename = f"training_log_epoch_{epoch}.json"
            file_path = save_dir / filename
            try:
                with open(file_path, 'w') as f:
                    json.dump(training_data, f, indent=4, cls=TensorEncoder)
                saved_log_files.append(file_path)
                logger.info(f'Saved training log to {file_path}')

                if len(saved_log_files) > 3:
                    oldest_file = saved_log_files.pop(0)
                    if oldest_file.exists():
                        oldest_file.unlink()
                        logger.info(f'Deleted old log file: {oldest_file}')
            except TypeError as e:
                logger.error(f"Error saving training log: {e}")

    # 最终测试集评估
    test_score, test_class_scores, test_loss, test_f1_macro, test_f1_per_class, test_iou, test_confusion_matrix = evaluate(
        net, val_loader, device, ampbool, traintype
    )
    
    print('\nFinal Test Results:')
    print(f'Test - Dice Score: {test_score:.4f}, F1 Score (macro): {test_f1_macro:.4f}, IoU: {test_iou:.4f}')
    print(f'Test - Class Dice Scores: {[f"{score:.4f}" for score in test_class_scores]}')
    print(f'Test - F1 Score per class: {[f"{f1:.4f}" for f1 in test_f1_per_class]}')
    print(f'Test Loss: {test_loss:.4f}')

    # 保存最终结果
    final_filename = "final_training_log.json"
    with open(save_dir / final_filename, 'w') as f:
        json.dump(training_data, f, indent=4, cls=TensorEncoder)

    # # 保存测试集混淆矩阵
    # if test_confusion_matrix is not None:
    #     save_confusion_matrix(test_confusion_matrix, str(save_dir), "final_test")

    writer.close()
    return net

def train_with_loqo(net,
                   device,
                   epochs: int = 30,          # 按论文修改为30
                   batch_size: int = 64,      # 按论文修改为64
                   learning_rate: float = 1e-5,
                   save_checkpoint: bool = True,
                   gradient_clipping: float = 1.0):
    """
    使用LOQO策略进行训练
    每次使用3/4的数据训练，1/4的数据测试
    """
    results = []
    
    # 对每个四分之一进行交叉验证
    for fold in range(4):
        print(f"Training fold {fold+1}/4")
        
        # 创建训练集（使用其他三个四分之一）
        train_dataset = EnhancedSatelliteDataset(
            pre_dir_img=train_data['pre_img'],
            pre_dir_mask=train_data['pre_mask'],
            post_dir_img=train_data['post_img'],
            post_dir_mask=train_data['post_mask'],
            patch_size=1024,
            stride=64,      # 实现960像素重叠
            augment=True,
            quarter_idx=-fold-1
        )
        
        # 创建测试集（使用当前四分之一）
        test_dataset = EnhancedSatelliteDataset(
            pre_dir_img=train_data['pre_img'],
            pre_dir_mask=train_data['pre_mask'],
            post_dir_img=train_data['post_img'],
            post_dir_mask=train_data['post_mask'],
            patch_size=1024,
            stride=1024,    # 测试时不需要重叠
            augment=False,
            quarter_idx=fold
        )
        
        # 创建数据加载器
        train_loader = DataLoader(train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                num_workers=4,    # 增加工作进程
                                pin_memory=True)  # 使用PIN_MEMORY加速
        test_loader = DataLoader(test_dataset, 
                               batch_size=1, 
                               shuffle=False)
        
        # 训练当前fold
        fold_results = train_net_enhanced(
            net=net,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_loader=train_loader,
            val_loader=test_loader,
            save_checkpoint=save_checkpoint,
            gradient_clipping=gradient_clipping
        )
        
        results.append(fold_results)
        
    return results

# def calculate_loss(pred, target):
#     """
#     仅在有标注的像素上计算交叉熵损失
#     """
#     # 创建掩码，标识有效的（已标注的）像素
#     valid_pixels = target != 255  # 假设255表示未标注
    
#     # 只在有效像素上计算损失
#     loss = F.cross_entropy(
#         pred[valid_pixels],
#         target[valid_pixels],
#         reduction='mean'
#     )
#     return loss

classes = 5
bilinear = True
# loadstate = False
loadstate = True
load = './training/checkpoints/v_1.3_lr_3.5e-05_20241104_010028/checkpoint_epoch60.pth'
# load = './checkpoints/best0921.pth'
start_epoch = 1
epochs = 60
batch_size = 4
# batch_size = 1
# lr = 2.69e-4
# lr = 8.125358e-4

lr = 3.5e-5
# gamma = 0.98
# lr = lr * (gamma ** 100)  # 计算经过100个epoch后的学习率

# lr = 1e-6
scale = 1
train = 1
# train =0.15259598603*2
val = 1
test = 1
ampbool = True
save_checkpoint = True
traintype = 'both'
gradclip = 1.0


if __name__ == '__main__':
    # task = Task.init(project_name="damage-assessment", task_name="enhanced_training_loqo")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = SiamUNetConCVgg19()
    # net = SiameseUNetWithResnet50Encoder()
    if loadstate:
        net.load_state_dict(torch.load(load, map_location=device))
        logger.info(f'Model loaded from {load}')

    net.to(device)
    
    # 使用LOQO训练策略
    results = train_with_loqo(
        net=net,
        device=device,
        epochs=30,          # 按论文设置
        batch_size=64,      # 按论文设置
        learning_rate=3.5e-5,
        save_checkpoint=True,
        gradient_clipping=1.0
    )