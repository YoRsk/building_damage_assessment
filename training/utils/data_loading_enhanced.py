from os import listdir
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2

class EnhancedSatelliteDataset(Dataset):
    """
    增强版卫星图像数据集，实现更复杂的数据增强策略
    """
    def __init__(self,
                 preimages_dir,
                 premasks_dir,
                 postimages_dir,
                 postmasks_dir,
                 patch_size=1024,
                 stride=512,  # 重叠步长
                 augment=True,
                 dilate_masks=True,
                 seed=42):
        """
        Args:
            patch_size: 图像块大小
            stride: 滑动窗口步长(用于生成重叠的patches)
            augment: 是否使用增强
            dilate_masks: 是否对mask进行膨胀处理
        """
        self.patch_size = patch_size
        self.stride = stride
        self.dilate_masks = dilate_masks
        
        # 基本路径设置
        self.postimages_dir = postimages_dir
        self.postmasks_dir = postmasks_dir
        self.preimages_dir = preimages_dir
        self.premasks_dir = premasks_dir
        
        # 获取图像ID列表
        self.ids = [file.split('_')[0:2] for file in listdir(postimages_dir) 
                   if not file.startswith('.')]
        
        # 创建增强pipeline
        if augment:
            self.transform = A.Compose([
                # 几何变换
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                ], p=0.5),
                
                A.ShiftScaleRotate(
                    shift_limit=0.0625,
                    scale_limit=0.1,
                    rotate_limit=45,
                    p=0.5
                ),
                
                # 光度变换 - 第一组
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                    A.ToGray(p=1.0),
                    A.ColorJitter(brightness=0.2, contrast=0.2),
                ], p=0.5),
                
                # 光度变换 - 第二组
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.RandomGamma(gamma_limit=(80, 120)),
                ], p=0.5),
                
                # 图像质量变换
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0),
                    A.GridDistortion(distort_limit=0.3),
                ], p=0.5),
                
            ], additional_targets={
                'image2': 'image',
                'mask2': 'mask'
            })
        else:
            self.transform = None

    def _generate_patches(self, image):
        """生成重叠的图像块"""
        patches = []
        h, w = image.shape[:2]
        
        for y in range(0, h-self.patch_size+1, self.stride):
            for x in range(0, w-self.patch_size+1, self.stride):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
        
        return patches

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        # 加载图像和掩码
        postmask_file = str(self.postmasks_dir) + '/' + name[0] + '_' + name[1] + '_post_disaster.png'
        premask_file = str(self.premasks_dir) + '/' + name[0] + '_' + name[1] + '_pre_disaster.png'
        postimg_file = str(self.postimages_dir) + '/' + name[0] + '_' + name[1] + '_post_disaster.png'
        preimg_file = str(self.preimages_dir) + '/' + name[0] + '_' + name[1] + '_pre_disaster.png'
        
        postmask = np.array(Image.open(postmask_file))
        postimg = np.array(Image.open(postimg_file))
        premask = np.array(Image.open(premask_file))
        preimg = np.array(Image.open(preimg_file))
        
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=postimg,
                mask=postmask,
                image2=preimg,
                mask2=premask
            )
            
            postimg = transformed['image']
            postmask = transformed['mask']
            preimg = transformed['image2']
            premask = transformed['mask2']
        
        # 对mask进行膨胀处理
        if self.dilate_masks:
            kernel = np.ones((3,3), np.uint8)
            postmask = cv2.dilate(postmask, kernel, iterations=1)
            premask = cv2.dilate(premask, kernel, iterations=1)
        
        # 转换为tensor
        return {
            'postimage': torch.from_numpy(postimg).permute(2, 0, 1).float() / 255.0,
            'postmask': torch.from_numpy(postmask).long(),
            'preimage': torch.from_numpy(preimg).permute(2, 0, 1).float() / 255.0,
            'premask': torch.from_numpy(premask).long()
        }

    def __len__(self):
        return len(self.ids) 