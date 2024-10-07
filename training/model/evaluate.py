import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.dice_score import multiclass_dice_coeff, dice_coeff, dice_loss
import torch.nn as nn
from torchmetrics import F1Score, JaccardIndex, ConfusionMatrix

def evaluate(net, dataloader, device, ampbool, traintype='post'):
    with torch.no_grad():
        net.eval()
        num_val_batches = len(dataloader)
        dice_score = 0
        dice_score_class = [0, 0, 0, 0, 0]
        criterion = nn.CrossEntropyLoss()
        epoch_loss = 0

        # 初始化指标
        f1_score = F1Score(task='multiclass', num_classes=5, average='macro').to(device)
        iou_score = JaccardIndex(task="multiclass", num_classes=5).to(device)
        confmat = ConfusionMatrix(num_classes=5).to(device)

        with tqdm(total=num_val_batches, desc='validation', unit='img') as pbar:
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
                        loss = criterion(mask_pred, true_masks)
                        loss += dice_loss(
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

                    # 更新指标
                    f1_score.update(mask_pred.argmax(dim=1), true_masks)
                    iou_score.update(mask_pred.argmax(dim=1), true_masks)
                    confmat.update(mask_pred.argmax(dim=1).flatten(), true_masks.flatten())

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

                    # 更新指标
                    f1_score.update(mask_pred.argmax(dim=1), true_masks)
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
        f1 = f1_score.compute()
        iou = iou_score.compute()
        confusion_matrix = confmat.compute()

        return [dice_score / num_val_batches, [i / num_val_batches for i in dice_score_class], epoch_loss, f1, iou, confusion_matrix]