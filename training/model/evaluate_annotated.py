import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import torch.nn as nn
from torchmetrics import F1Score, JaccardIndex, ConfusionMatrix
import logging

logger = logging.getLogger(__name__)

def evaluate_annotated(net, dataloader, device, ampbool, traintype='both'):
    num_val_batches = len(dataloader)
    total_val_samples = len(dataloader.dataset)
    no_annotation_count = 0
    
    with torch.no_grad():
        net.eval()
        dice_score = 0
        dice_score_class = [0, 0, 0, 0, 0]
        criterion = nn.CrossEntropyLoss(reduction='none')
        epoch_loss = 0
        valid_batches = 0

        f1_score_per_class = F1Score(task='multiclass', num_classes=5, average=None).to(device)
        iou_score = JaccardIndex(task="multiclass", num_classes=5).to(device)
        
        with tqdm(total=total_val_samples, desc='Validation', unit='img') as pbar:
            if traintype == 'both':
                for batch_idx, batch in enumerate(dataloader):
                    preimage, postimage, true_masks = batch['preimage'], batch['postimage'], batch['postmask']

                    preimage = preimage.to(device=device, dtype=torch.float32)
                    postimage = postimage.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)

                    with torch.amp.autocast('cuda', enabled=ampbool):
                        mask_pred = net(preimage, postimage)
                        pred_classes = mask_pred.argmax(dim=1)
                        
                        # 更新指标（在所有像素上）
                        f1_score_per_class.update(pred_classes, true_masks)
                        iou_score.update(pred_classes, true_masks)

                        # 计算 Dice 分数（在所有像素上）
                        mask_pred_onehot = F.one_hot(pred_classes, 5).permute(0, 3, 1, 2).float()
                        true_onehot = F.one_hot(true_masks, 5).permute(0, 3, 1, 2).float()
                        
                        dice_score += multiclass_dice_coeff(
                            mask_pred_onehot[:, 1:, ...],
                            true_onehot[:, 1:, ...],
                            reduce_batch_first=False
                        )
                        
                        # 计算每个类别的 Dice score
                        for i in range(5):
                            dice_score_class[i] += dice_coeff(
                                mask_pred_onehot[:, i, ...],
                                true_onehot[:, i, ...],
                                reduce_batch_first=False
                            )

                        # 获取已标注像素的掩码（仅用于损失计算）
                        annotated_pixels = true_masks > 0
                        if annotated_pixels.any():
                            valid_batches += 1
                            
                            # 计算交叉熵损失（只在已标注像素上）
                            pixel_losses = criterion(mask_pred, true_masks)
                            ce_loss = pixel_losses[annotated_pixels].mean()
                            
                            # 计算 Dice Loss（只在已标注像素上）
                            pred_softmax = F.softmax(mask_pred, dim=1).float()
                            pred_softmax_masked = pred_softmax[:, 1:, ...].clone()
                            true_masks_onehot = true_onehot[:, 1:, ...]
                            
                            # 将未标注像素的预测和真实��都设为0（仅用于损失计算）
                            for i in range(pred_softmax_masked.shape[1]):
                                pred_softmax_masked[:, i, ...][~annotated_pixels] = 0
                                true_masks_onehot[:, i, ...][~annotated_pixels] = 0
                            
                            dice = dice_loss(
                                pred_softmax_masked,
                                true_masks_onehot,
                                multiclass=True
                            )
                            
                            # 合并损失
                            loss = ce_loss + dice
                            epoch_loss += loss.item()
                        else:
                            no_annotation_count += 1
                            logger.warning(f"验证批次 {batch_idx} 没有标注像素！")
                            print(f"Warning: 验证批次 {batch_idx} 没有标注像素！")

                    pbar.update(postimage.shape[0])

        # 使用总批次数进行平均（因为���标在所有像素上计算）
        dice_score = dice_score / num_val_batches
        dice_score_class = [i / num_val_batches for i in dice_score_class]
        
        # 使用有效批次数进行损失平均
        valid_batches = max(valid_batches, 1)  # 避免除以零
        epoch_loss = epoch_loss / valid_batches
        
        # 计算最终指标
        f1_per_class = f1_score_per_class.compute()
        f1_macro = f1_per_class.mean()
        iou = iou_score.compute()

        net.train()

        return [dice_score,
                dice_score_class, 
                epoch_loss,
                f1_macro, 
                f1_per_class, 
                iou] 