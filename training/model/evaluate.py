import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import torch.nn as nn
from torchmetrics import F1Score, JaccardIndex, ConfusionMatrix
from torchmetrics import AUROC

def evaluate(net, dataloader, device, ampbool, traintype='post'):
    num_val_batches = len(dataloader)
    total_val_samples = len(dataloader.dataset)  # 获取验证集的总样本数
    with torch.no_grad():
        net.eval()
        num_val_batches = len(dataloader)
        dice_score = 0
        dice_score_class = [0, 0, 0, 0, 0]
        criterion = nn.CrossEntropyLoss()
        
        epoch_loss = 0

        # 只初始化一个 F1Score 指标，用于计算每个类别的 F1 分数
        f1_score_per_class = F1Score(task='multiclass', num_classes=5, average=None).to(device)
        iou_score = JaccardIndex(task="multiclass", num_classes=5).to(device)
        # normalize='true' 在真实标签做归一化
        confmat = ConfusionMatrix(task="multiclass", num_classes=5, normalize='true').to(device) 
        # auroc = AUROC(task="multiclass", num_classes=5).to(device) 
        with tqdm(total=total_val_samples, desc='Validation', unit='img') as pbar:
            if (traintype == 'both'):
                loss = 0
                for batch in dataloader:
                    preimage, postimage, true_masks = batch['preimage'], batch['postimage'], batch['postmask']

                    preimage = preimage.to(device=device, dtype=torch.float32)
                    postimage = postimage.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    mask_true = F.one_hot(true_masks, 5).permute(0, 3, 1, 2).float()
                    with torch.amp.autocast('cuda', enabled=ampbool):
                        # predict the mask
                        mask_pred = net(preimage, postimage)
                        # convert to one-hot format
                        # 暂时不用交叉熵
                        # loss = criterion(mask_pred, true_masks)
                        loss = dice_loss(
                            F.softmax(mask_pred, dim=1).float()[:, 1:, ...],
                            F.one_hot(true_masks, 5).permute(0, 3, 1, 2).float()[:, 1:, ...],
                            multiclass=True
                        )
                        if 5 == 1:
                            mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                            # compute the Dice score
                            dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                        else:
                            mask_pred = F.one_hot(mask_pred.argmax(dim=1), 5).permute(0, 3, 1, 2).float()
                            # compute the Dice score, ignoring background
                            dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                                reduce_batch_first=False)
                            # 计算每个类别的 Dice 系数 并且包含unclassified
                            dice_score_class[0] += dice_coeff(mask_pred[:, 0, ...], mask_true[:, 0, ...],
                                                            reduce_batch_first=False)
                            dice_score_class[1] += dice_coeff(mask_pred[:, 1, ...], mask_true[:, 1, ...],
                                                            reduce_batch_first=False)
                            dice_score_class[2] += dice_coeff(mask_pred[:, 2, ...], mask_true[:, 2, ...],
                                                            reduce_batch_first=False)
                            dice_score_class[3] += dice_coeff(mask_pred[:, 3, ...], mask_true[:, 3, ...],
                                                            reduce_batch_first=False)
                            dice_score_class[4] += dice_coeff(mask_pred[:, 4, ...], mask_true[:, 4, ...],
                                                            reduce_batch_first=False)
                            
                    f1_score_per_class.update(mask_pred.argmax(dim=1), true_masks)
                    iou_score.update(mask_pred.argmax(dim=1), true_masks)
                    confmat.update(mask_pred.argmax(dim=1).flatten(), true_masks.flatten())
                    # auroc.update(F.softmax(mask_pred, dim=1), true_masks)

                    pbar.update(postimage.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    epoch_loss += loss.item()
            if (traintype == 'post'):
                for batch in dataloader:
                    postimage, true_masks = batch['postimage'], batch['postmask']

                    postimage = postimage.to(device=device, dtype=torch.float32)
                    true_masks = true_masks.to(device=device, dtype=torch.long)
                    mask_true = F.one_hot(true_masks, 5).permute(0, 3, 1, 2).float()
                    with torch.amp.autocast('cuda', enabled=ampbool):
                        # predict the mask
                        mask_pred = net(postimage)
                        # convert to one-hot format
                        loss = criterion(mask_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(mask_pred, dim=1).float()[:, 1:, ...],
                            F.one_hot(true_masks, 5).permute(0, 3, 1, 2).float()[:, 1:, ...],
                            multiclass=True
                        )
                        mask_pred = F.one_hot(mask_pred.argmax(dim=1), 5).permute(0, 3, 1, 2).float()
                        # compute the Dice score, ignoring background
                        dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                            reduce_batch_first=False)

                    # 在更新指标时，保持原来的逻辑
                    f1_score_per_class.update(mask_pred.argmax(dim=1), true_masks)
                    iou_score.update(mask_pred.argmax(dim=1), true_masks)
                    confmat.update(mask_pred.argmax(dim=1).flatten(), true_masks.flatten())

                    pbar.update(postimage.shape[0])
                    pbar.set_postfix(**{'loss (batch)': loss.item()})
                    epoch_loss += loss.item()

        net.train()

        # Fixes a potential division by zero error
        if num_val_batches == 0:
            return [dice_score, dice_score_class, epoch_loss, 0, 0, None]

        # 计算最终的指标值
        f1_per_class = f1_score_per_class.compute()
        f1_macro = f1_per_class.mean()  # 计算宏平均 F1 分数
        iou = iou_score.compute()
        confusion_matrix = confmat.compute()
        # auc_roc = auroc.compute()  # 添加这行

        # 移除未分类类别的结果
        confusion_matrix = confusion_matrix[1:, 1:]

        return [dice_score / num_val_batches, [i / num_val_batches for i in dice_score_class], 
                epoch_loss, f1_macro, f1_per_class, iou, confusion_matrix]