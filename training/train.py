from model.my_models import SiameseUNetWithResnet50Encoder
# from model.cbam_res50 import ResNet50
from model.evaluate import evaluate
from utils.data_loading import SatelliteDataset
from utils.dice_score import dice_loss

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses.focal import FocalLoss
from torchmetrics import ConfusionMatrix

import math
import numpy as np
from tqdm import tqdm   
from pathlib import Path


import torch.optim as optim

import torchmetrics
import logging
from pathlib import Path
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

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 创建一个文件处理器
file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.DEBUG)

# 创建一个格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将处理器添加到 logger
logger.addHandler(file_handler)
img_home_path = "C:/Users/xiao/peng/xbd/Dataset"
# img_home_path = "C:/Users/xiao/peng/xbd/Dataset_test"
# img_home_path = "C:/Users/liuyi/segment/ubdd/xbd/Dataset_test"
post_dir_img = img_home_path + "/TierFull/Post/Image512/"
post_dir_mask = img_home_path + "/TierFull/Post/Label512/"
pre_dir_img = img_home_path + "/TierFull/Pre/Image512/"
pre_dir_mask = img_home_path + "/TierFull/Pre/Label512/"
post_dir_val_img = img_home_path + "/Validation/Post/Image512/"
post_dir_val_mask = img_home_path + "/Validation/Post/Label512/"
pre_dir_val_img = img_home_path + "/Validation/Pre/Image512/"
pre_dir_val_mask = img_home_path + "/Validation/Pre/Label512/"
post_dir_test_img = img_home_path + "/Test/Post/Image512/"
post_dir_test_mask = img_home_path + "/Test/Post/Label512/"
pre_dir_test_img = img_home_path + "/Test/Pre/Image512/"
pre_dir_test_mask = img_home_path + "/Test/Pre/Label512/"
# post_dir_img = Path('.\\Dataset\\TierFull\\Post\\Image512\\')
# post_dir_mask = Path('.\\Dataset\\TierFull\\Post\\Label512\\')
# pre_dir_img = Path('.\\Dataset\\TierFull\\Pre\\Image512\\')
# pre_dir_mask = Path('.\\Dataset\\TierFull\\Pre\\Label512\\')
# post_dir_val_img = Path('.\\Dataset\\Validation\\Post\\Image512\\')
# post_dir_val_mask = Path('.\\Dataset\\Validation\\Post\\Label512\\')
# pre_dir_val_img = Path('.\\Dataset\\Validation\\Pre\\Image512\\')
# pre_dir_val_mask = Path('.\\Dataset\\Validation\\Pre\\Label512\\')

# 创建带有时间戳的 checkpoint 目录
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# dir_checkpoint = Path(f'checkpoints/v_1.3_lr_8.1e-4_{timestamp}/')
# dir_checkpoint = Path('checkpoints/v_1.2_lr_5e-4_cbam/')
# dir_checkpoint = Path('checkpoints/v_1.0_lr_10-6/')

closs = nn.CrossEntropyLoss()

floss = FocalLoss(mode = 'multiclass',
                alpha = None,
                gamma = 2.0,
                ignore_index = None,
                reduction = "mean",
                normalized = False,
                reduced_threshold = None)
# ########混淆矩阵 （不行就删除）
# #类别名字
# class_names = {
#     0: "no-damage",
#     1: "minor-damage",
#     2: "major-damage",
#     3: "destroyed"
# }

def save_confusion_matrix(confusion_matrix, save_dir, filename_prefix):
    # 确保 confusion_matrix 是 numpy 数组
    if isinstance(confusion_matrix, torch.Tensor):
        confusion_matrix = confusion_matrix.cpu().numpy()

    # # 移除 "unclassified" 类别（假设它是第一行和第一列）
    # confusion_matrix = confusion_matrix[1:, 1:]

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 保存原始混淆矩阵为numpy数组
    np.save(os.path.join(save_dir, f"{filename_prefix}_confusion_matrix.npy"), confusion_matrix)

    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='.3f', cmap='Blues',
                xticklabels=range(0, 5),  # 使用0 1、2、3, 4作为标签
                yticklabels=range(0, 5))  # 使用0 1、2、3, 4作为标签
    plt.title('Confusion Matrix in Damage Level')
    # 保存图片
    plt.savefig(os.path.join(save_dir, f"{filename_prefix}_confusion_matrix.png"))
    plt.close()
###############
def train_net(net,
              device,
              start_epoch: int = 1,
              epochs: int = 5,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              train_percent: float = 1,
              val_percent: float = 1,
              test_percent: float = 1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              ampbool: bool = False,
              traintype: str = 'both',
              gradient_clipping: float = 1.0):

    # 创建带有版本号、学习率和时间戳的目录名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f'v_1.3_lr_{learning_rate:.1e}_{timestamp}'
    
    # 创建checkpoints目录
    dir_checkpoint = Path(f'checkpoints/{dir_name}/')
    dir_checkpoint.mkdir(parents=True, exist_ok=True)
    
    # 创建training_logs目录
    save_dir = Path(f'training_logs/{dir_name}/')
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建 TensorBoard 日志目录
    log_dir = Path(f'tensorboard_logs/{dir_name}/')
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))


    # 创建一个字典来存储训练参数和结果
    training_data = {
        "hyperparameters": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "img_scale": img_scale,
            "train_percent": train_percent,
            "val_percent": val_percent,
            "ampbool": ampbool,
            "traintype": traintype,
            "gradient_clipping": gradient_clipping
        },
        "training_history": []
    }


    # 1. Create dataset
    train = None
    validate = None
    test_set = None
    try:
        train = SatelliteDataset(pre_dir_img, pre_dir_mask, post_dir_img, post_dir_mask, 1, values=[[.8,1.5], True, True, True], probabilities=[0, 0, 0, 0])
        validate = SatelliteDataset(pre_dir_val_img, pre_dir_val_mask, post_dir_val_img, post_dir_val_mask, 1, values=[[1, 1], False, False, False], probabilities=[0, 0, 0, 0])
        test_set = SatelliteDataset(pre_dir_test_img, pre_dir_test_mask, post_dir_test_img, post_dir_test_mask, 1, values=[[1, 1], False, False, False], probabilities=[0, 0, 0, 0])
    except (AssertionError, RuntimeError):
        print('Error creating datasets')

    loader_args = dict(batch_size=batch_size, num_workers=1, pin_memory=True)
    
    # 2. Split into train / validation partitions
    n_train = int(len(train) * train_percent)
    n_train_none = int(len(train) - n_train)
    n_val = int(len(validate) * val_percent)
    n_val_none = int(len(validate) - n_val)
    n_test = int(len(test_set) * test_percent)
    n_test_none = int(len(test_set) - n_test)

    train_set, train_val_none_set = random_split(train, [n_train, n_train_none], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)

    val_set, val_none_set = random_split(validate, [n_val, n_val_none], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    
    test_set, test_none_set = random_split(test_set, [n_test, n_test_none], generator=torch.Generator().manual_seed(0))
    test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **loader_args)

########################################################
   # 在创建数据集和数据加载器之后，添加以下打印语句

    # print(f"Total training samples: (train){len(train)}")
    # print(f"Total validation samples: (validate){len(validate)}")
    # print(f"n_train: {n_train}")
    # print(f"n_val: {n_val}")
    # print(f"Number of batches in train_loader: {len(train_loader)}")
    # print(f"Number of batches in val_loader: {len(val_loader)}")
#########################################################

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
      #optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-6, momentum=0.9, foreach=True)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)    
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,11,17,23,29,33,55,78,100], gamma=0.30)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=32, verbose=True)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, verbose=True)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,9,12,15,18,19,20,33,47,50,60,70,90,110,130,150,170,180,190], gamma=0.5)
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=ampbool)
    criterion = closs
    
    # 5. Initialize torchmetrics metrics
    train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=5, validate_args=False).to(device)
    train_precision = torchmetrics.Precision(task='multiclass', num_classes=5, average='macro', validate_args=False).to(device)
    train_recall = torchmetrics.Recall(task='multiclass', num_classes=5, average='macro', validate_args=False).to(device)
    train_f1 = torchmetrics.F1Score(task='multiclass', num_classes=5, average='macro').to(device)
    train_iou = JaccardIndex(task="multiclass", num_classes=5).to(device)
    
    #TODO: 是否加入早停
    #early_stopping = EarlyStopping(patience=3, verbose=True)
    # 5.1. 获取 ClearML 任务对象
    task = Task.current_task()
    if task is None:
        print("Warning: No active ClearML task. Metrics will not be logged.")

    # 6. Begin training
    saved_checkpoints = []
    saved_confusion_matrices = []
    
    # 用于跟踪已保存的日志文件
    saved_log_files = []

    for epoch in range(start_epoch, start_epoch + epochs):
        net.train()
        epoch_loss = 0
        epoch_steps = 0 
        nancount = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
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
                        # 暂时不用交叉熵
                        # loss = criterion(masks_pred, post_masks)
                        loss = dice_loss(
                            F.softmax(masks_pred, dim=1).float()[:, 1:, ...],
                            F.one_hot(post_masks, 5).permute(0, 3, 1, 2).float()[:, 1:, ...],
                            multiclass=True
                        )
                    # 可能用不了
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
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                # Update metrics
                train_accuracy.update(masks_pred, post_masks)
                train_precision.update(masks_pred, post_masks)
                train_recall.update(masks_pred, post_masks)
                train_f1.update(masks_pred, post_masks)
                train_iou.update(masks_pred.argmax(dim=1), post_masks)
                
                pbar.update(postimage.shape[0])
                epoch_steps += 1 #已经处理的批次数
                
                if math.isnan(loss.item()):
                    epoch_loss += 0
                    nancount += 1
                else:
                    epoch_loss += loss.item()   
                    
                # if epoch_steps == 1 or epoch_steps % max(1, len(train_loader) // 10) == 0:
                pbar.set_postfix(**{
                    'loss (batch)': float(loss.item()),
                    'loss': float(epoch_loss / epoch_steps),
                    'accuracy': float(train_accuracy.compute().item()),
                    'precision': float(train_precision.compute().item()),
                    'recall': float(train_recall.compute().item()),
                    'f1_score': float(train_f1.compute().item()),
                    'iou': float(train_iou.compute().item())
                })
                
        # 计算最终的训练指标
        train_acc = train_accuracy.compute()
        train_prec = train_precision.compute()
        train_rec = train_recall.compute()
        train_f1_score = train_f1.compute()
        train_iou_score = train_iou.compute()
        train_loss = epoch_loss / len(train_loader)
        
        # 使用现有的evaluate函数进行验证
        val_score, val_class_scores, val_loss, val_f1_macro, val_f1_per_class, val_iou, val_confusion_matrix = evaluate(net, dataloader=val_loader, device=device, ampbool=ampbool, traintype=traintype)
        # 获取当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        # 更新学习率
        scheduler.step()

        # 记录到本地文件
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
            "learning_rate": current_lr,
            # "val_auc_roc": val_auc_roc.item(),
        }
        training_data["training_history"].append(epoch_data)


        # 记录到 TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_score, epoch)
        writer.add_scalar('Precision/train', train_prec, epoch)
        writer.add_scalar('Recall/train', train_rec, epoch)
        writer.add_scalar('F1 Score/train', train_f1_score, epoch)
        writer.add_scalar('F1 Score/validation_macro', val_f1_macro, epoch)
        for i, f1 in enumerate(val_f1_per_class):
            writer.add_scalar(f'F1 Score/validation_class_{i}', f1, epoch)
        writer.add_scalar('IoU/train', train_iou_score, epoch)
        writer.add_scalar('IoU/validation', val_iou, epoch)
        # writer.add_scalar('AUC-ROC/validation', val_auc_roc, epoch)
        for i, class_score in enumerate(val_class_scores):
            writer.add_scalar(f'Dice Score/class_{i}', class_score, epoch)
        ############


        # 打印训练和验证结果
        print(f'Epoch {epoch}')
        print(f'Training - Accuracy: {train_acc:.4f}, Precision: {train_prec:.4f}, Recall: {train_rec:.4f}, F1 Score: {train_f1_score:.4f}, IoU: {train_iou_score:.4f}')
        print(f'Training Loss: {train_loss:.4f}')
        print(f'Validation - Dice Score: {val_score:.4f}, F1 Score (macro): {val_f1_macro:.4f}, IoU: {val_iou:.4f}')
        print(f'Validation - F1 Score per class: {[f"{f1:.4f}" for f1 in val_f1_per_class]}')
        print(f'Validation - Class Dice Scores: {[f"{score:.4f}" for score in val_class_scores]}')
        print(f'Validation Loss: {val_loss:.4f}')
        # print(f'Validation - AUC-ROC: {val_auc_roc:.4f}')
        print(f'NaN Count: {nancount}')
        
        # 保存checkpoint
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            checkpoint_path = dir_checkpoint / f'checkpoint_epoch{epoch}.pth'
            torch.save(net.state_dict(), str(checkpoint_path))
            saved_checkpoints.append(str(checkpoint_path))
            logger.info(f'Checkpoint {epoch} saved!')
            # 删除旧的检查点
            if len(saved_checkpoints) > 10:
                old_checkpoint = Path(saved_checkpoints.pop(0))
                if old_checkpoint.exists():
                    old_checkpoint.unlink()
                    logger.info(f'Deleted old checkpoint: {old_checkpoint}')
            
        # 保存混淆矩阵
        if val_confusion_matrix is not None:
            cm_path = save_dir / f"epoch_{epoch}_validation_confusion_matrix"
            save_confusion_matrix(val_confusion_matrix, str(save_dir), f"epoch_{epoch}_validation")
            saved_confusion_matrices.append(str(cm_path))
            logger.info(f'Confusion Matrix {epoch} saved!')
            # 删除旧的混淆矩阵
            if len(saved_confusion_matrices) > 10:
                old_cm = Path(saved_confusion_matrices.pop(0))
                if (old_cm.with_suffix('.png')).exists():
                    (old_cm.with_suffix('.png')).unlink()
                if (old_cm.with_suffix('.npy')).exists():
                    (old_cm.with_suffix('.npy')).unlink()
                logger.info(f'Deleted old confusion matrix: {old_cm}')
        else:
            logger.info("val_confusion_matrix is None")
        # 保存到 ClearML
        if task:
             task.get_logger().report_scalar("Accuracy", "train", value=train_acc, iteration=epoch)
             task.get_logger().report_scalar("Precision", "train", value=train_prec, iteration=epoch)
             task.get_logger().report_scalar("Recall", "train", value=train_rec, iteration=epoch)
             task.get_logger().report_scalar("F1 Score", "train", value=train_f1_score, iteration=epoch)
             task.get_logger().report_scalar("Loss", "train", value=train_loss, iteration=epoch)
             task.get_logger().report_scalar("Dice Score", "validation", value=val_score, iteration=epoch)
             task.get_logger().report_scalar("F1 Score", "validation_macro", value=val_f1_macro, iteration=epoch)
             for i, f1 in enumerate(val_f1_per_class):
                 task.get_logger().report_scalar(f"F1 Score Class {i}", "validation", value=f1, iteration=epoch)
             task.get_logger().report_scalar("Loss", "validation", value=val_loss, iteration=epoch)
             task.get_logger().report_scalar("Learning Rate", "", value=current_lr, iteration=epoch)
             task.get_logger().report_scalar("IoU", "train", value=train_iou_score, iteration=epoch)
             task.get_logger().report_scalar("IoU", "validation", value=val_iou, iteration=epoch)
            #  task.get_logger().report_scalar("AUC-ROC", "validation", value=val_auc_roc, iteration=epoch)
        # 定期保存training_data到log文件
        if epoch % 5 == 0 or epoch == epochs - 1:
            filename = f"training_log_epoch_{epoch}.json"
            file_path = save_dir / filename
            try: 
                with open(file_path, 'w') as f:
                    json.dump(training_data, f, indent=4, cls=TensorEncoder)
                
                saved_log_files.append(file_path)
                logger.info(f'Saved training log to {file_path}')

                # 如果保存的文件数量超过3个,删除最旧的文件
                if len(saved_log_files) > 3:
                    oldest_file = saved_log_files.pop(0)
                    if oldest_file.exists():
                        oldest_file.unlink()
                        logger.info(f'Deleted old log file: {oldest_file}')

            except TypeError as e:
                logger.error(f"捕获到异常：{e}")

        # # 保存混淆矩阵
        # if val_confusion_matrix is not None:
        #     save_confusion_matrix(val_confusion_matrix, save_dir, f"epoch_{epoch}_validation")

    # Final evaluation on test set
    test_score, test_class_scores, test_loss, test_f1_macro, test_f1_per_class, test_iou, test_confusion_matrix= evaluate(net, test_loader, device, ampbool, traintype)
    print('Final Test Results:')
    print(f'Test - Dice Score: {test_score:.4f}, F1 Score (macro): {test_f1_macro:.4f}, IoU: {test_iou:.4f}')
    print(f'Test - Class Dice Scores: {[f"{score:.4f}" for score in test_class_scores]}')
    print(f'Test - F1 Score per class: {[f"{f1:.4f}" for f1 in test_f1_per_class]}')
    print(f'Test Loss: {test_loss:.4f}')
    # print(f'Test - AUC-ROC: {test_auc_roc:.4f}')
    ########
    # 保存最终结果
    final_filename = f"final_training_log.json"
    with open(save_dir / final_filename, 'w') as f:
        json.dump(training_data, f, indent=4, cls=TensorEncoder)

    # 关闭 TensorBoard 写入器
    writer.close()
    ########

    # 保存测试集的混淆矩阵
    if test_confusion_matrix is not None:
        save_confusion_matrix(test_confusion_matrix, str(save_dir), "final_test")

    # # 训练循环结束后
    # while len(saved_checkpoints) > 10:
    #     old_checkpoint = Path(saved_checkpoints.pop(0))
    #     if old_checkpoint.exists():
    #         old_checkpoint.unlink()
    #         logger.info(f'Deleted old checkpoint: {old_checkpoint}')

    # while len(saved_confusion_matrices) > 10:
    #     old_cm = Path(saved_confusion_matrices.pop(0))
    #     if (old_cm.with_suffix('.png')).exists():
    #         (old_cm.with_suffix('.png')).unlink()
    #     if (old_cm.with_suffix('.npy')).exists():
    #         (old_cm.with_suffix('.npy')).unlink()
    #     logger.info(f'Deleted old confusion matrix: {old_cm}')

    return net


classes = 5
bilinear = True
loadstate = False
# loadstate = True
# load = './checkpoints/v_1.2_lr_5e-4/best.pth'
# load = './checkpoints/Vgg19SiamConc/checkpoint_epoch12.pth'
start_epoch = 1
# start_epoch = 13
# epochs = 100
epochs = 50
# epochs = 11
batch_size = 4
# batch_size = 1
# lr = 2.69e-4
# lr = 8.125358e-4
lr = 1.38e-4
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
# net = SiamUNetConCVgg19()
net = SiameseUNetWithResnet50Encoder()
# net = ResNet50()
model_parameters = filter(lambda p: p.requires_grad, net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
mem_params = sum([param.nelement()*param.element_size() for param in net.parameters()])
mem_bufs = sum([buf.nelement()*buf.element_size() for buf in net.buffers()])
mem = mem_params + mem_bufs

if __name__ == '__main__':
    task = Task.init(project_name="damage-assessment", task_name="train SiameseUNetWithResnet50Encoder 0920")

    # task_id_to_resume = "432ff2e399124e32977edbeb13c7e30a"  # 替换为您想恢复的任务 ID

    # # 初始化 ClearML Task
    # task = Task.get_task(task_id=task_id_to_resume) 
    # task.mark_started(force=True)
    # # 记录超参数
    task.connect({
         "batch_size": batch_size,
         "learning_rate": lr,
         "epochs": epochs,
         "img_scale": scale,
         "train_percent": train,
         "val_percent": val,
         "ampbool": ampbool,
         "traintype": traintype,
         "gradient_clipping": gradclip
     })
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    #net = SiamUNetDiff()
    #net = SiamUNetConC()
    #net = SiameseUNetWithResnet50Encoder()
    #net = SiamUnet_diff_Full(3,5)
    #net = UNetWithResnet50Encoder(5,'custom','./checkpoints/Resnet50ContrastiveWeight/checkpoint_epoch0.pth')
    #net = UNet(out_classes=classes, up_sample_mode='conv_transpose',batch_norm=False)
    #net = SiameseUNet(n_channels=3, n_classes = classes, bilinear=bilinear)
    #net = UNetVgg19V2(out_classes=classes, up_sample_mode='conv_transpose')
    #net = vgg19bn_unet(5,True)
    #net = vgg19nobn_unet(5,True)
    #net = vgg19nobn_unetdouble(5,True)
    #net = VGGUnet19nobnspace(out_channels = 5, pretrained = True)
    #net = SiamUNetCombVgg19()
    ### 之前是用这个 net = SiamUNetConCVgg19()
    #net = resnet_unet()
    #net = VGG16Unet(out_channels = 5)
    #net = resnet_siamunet()
    #net = SiamUNetDiffVgg19()
    #net = SiamUNetFullConCVgg19()
    #net = SiamUNetConCResnet50(True,5)
    #net = UNETResnet50(True,2)
    '''net = smp.Unet(
        encoder_name='resnext101_32x8d',
        encoder_depth=5,
        encoder_weights='imagenet',
        decoder_use_batchnorm=False,
        decoder_channels=(1024,512,256,128, 64),
        decoder_attention_type=None,
        in_channels=3,
        classes=5,
        activation=None,
        aux_params=None
    )
    '''
    if loadstate:
        net.load_state_dict(torch.load(load, map_location=device))
        logger.info(f'Model loaded from {load}')

    net.to(device=device)

    train_net(net=net,
                  start_epoch = start_epoch,
                  epochs=epochs,
                  batch_size=batch_size,
                  learning_rate=lr,
                  device=device,
                  img_scale=scale,
                  train_percent = train,
                  val_percent=val,
                  save_checkpoint = save_checkpoint,
                  ampbool = ampbool,
                  traintype = traintype,
                  gradient_clipping=gradclip)