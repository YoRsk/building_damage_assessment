from os import listdir
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
import cv2
import random

class EnhancedSatelliteDataset(Dataset):
    """
    增强版卫星图像数据集，实现更复杂的数据增强策略
    """
    def __init__(self,
                 pre_dir_img,
                 pre_dir_mask,
                 post_dir_img,
                 post_dir_mask,
                 patch_size=1024,
                 stride=64,
                 augment=True,
                 quarter_idx=None):
        """
        Args:
            pre_dir_img: pre-war图像目录
            pre_dir_mask: pre-war掩码目录
            post_dir_img: post-war图像目录
            post_dir_mask: post-war掩码目录
            patch_size: 图像块大小
            stride: 滑动窗口步长（64表示960像素重叠）
            augment: 是否使用增强
            quarter_idx: LOQO的四分之一索引(0-3)，None表示使用全图
        """
        self.pre_dir_img = pre_dir_img
        self.pre_dir_mask = pre_dir_mask
        self.post_dir_img = post_dir_img
        self.post_dir_mask = post_dir_mask
        self.patch_size = patch_size
        self.stride = stride
        
        # 只存储文件名配对
        pre_files = sorted([f for f in listdir(pre_dir_img) if not f.startswith('.')])
        post_files = set(listdir(post_dir_img))
        
        self.image_pairs = []
        for pre_file in pre_files:
            base_name = pre_file.replace('_pre.tif', '')
            post_file = f"{base_name}_post.tif"
            if post_file in post_files:
                self.image_pairs.append((pre_file, post_file))
        
        print(f"Found {len(self.image_pairs)} paired images")
        
        # 设置数据增强
        if augment:
            self.transform = A.Compose([
                # 几何变换 (50%概率)
                A.OneOf([
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Compose([
                        A.HorizontalFlip(p=1),
                        A.VerticalFlip(p=1)
                    ], p=0.5)
                ], p=0.5),
                
                # 仿射变换 (50%概率)
                A.ShiftScaleRotate(
                    shift_limit=0.0625,  # [-0.0625, 0.0625]
                    scale_limit=0.1,     # [0.9, 1.1]
                    rotate_limit=45,     # [-45°, 45°]
                    p=0.5
                ),
                
                # 光度变换 - 第一组 (50%概率)
                A.OneOf([
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                    A.ToGray(p=1.0),
                    A.ColorJitter(p=1.0)
                ], p=0.5),
                
                # 光度变换 - 第二组 (50%概率)
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=0.2,
                        contrast_limit=0.2
                    ),
                    A.RandomGamma(gamma_limit=(80, 120))
                ], p=0.5),
                
                # 图像质量变换 - 第三组 (50%概率)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.Downscale(scale_min=0.25, scale_max=0.25, interpolation=0),
                    A.GridDistortion(distort_limit=0.3)
                ], p=0.5),
                
            ], additional_targets={
                'image2': 'image',
                'mask2': 'mask'
            })
        else:
            self.transform = None
            
        # 如果指定了quarter_idx，进行LOQO划分
        if quarter_idx is not None:
            self.quarter_idx = quarter_idx
            self.quarters_to_use = self._get_quarters_to_use(quarter_idx)

    def _get_quarters_to_use(self, quarter_idx):
        """确定使用哪些quarters"""
        if quarter_idx < 0:  # 训练集：使用3个quarters
            return [i for i in range(4) if i != abs(quarter_idx) - 1]
        else:  # 测试集：使用1个quarter
            return [quarter_idx]

    def _dilate_mask(self, mask):
        """
        对建筑物掩码进行3x3膨胀处理
        """
        kernel = np.ones((3,3), np.uint8)
        return cv2.dilate(mask, kernel, iterations=1)

    def __getitem__(self, idx):
        pre_name, post_name = self.image_pairs[idx]
        
        # 加载并确保是RGB模式
        pre_img = Image.open(Path(self.pre_dir_img) / pre_name).convert('RGB')
        post_img = Image.open(Path(self.post_dir_img) / post_name).convert('RGB')
        pre_mask = Image.open(Path(self.pre_dir_mask) / pre_name).convert('L')  # 单通道掩码
        post_mask = Image.open(Path(self.post_dir_mask) / post_name).convert('L')
        
        # 转换为numpy数组
        pre_img = np.array(pre_img)
        post_img = np.array(post_img)
        pre_mask = np.array(pre_mask)
        post_mask = np.array(post_mask)
        
        # 对掩码进行膨胀处理
        pre_mask = self._dilate_mask(pre_mask)
        post_mask = self._dilate_mask(post_mask)
        
        # 分割图像为quarters
        h, w = pre_img.shape[:2]
        mid_h, mid_w = h//2, w//2
        
        quarters = {
            0: (slice(0, mid_h), slice(0, mid_w)),      # 左上
            1: (slice(0, mid_h), slice(mid_w, w)),      # 右上
            2: (slice(mid_h, h), slice(0, mid_w)),      # 左下
            3: (slice(mid_h, h), slice(mid_w, w))       # 右下
        }
        
        # 根据LOQO策略提取patches
        patches = []
        for q in self.quarters_to_use:
            h_slice, w_slice = quarters[q]
            pre_patch = pre_img[h_slice, w_slice]
            post_patch = post_img[h_slice, w_slice]
            pre_mask_patch = pre_mask[h_slice, w_slice]
            post_mask_patch = post_mask[h_slice, w_slice]
            
            # 使用stride创建重叠的patches
            for i in range(0, pre_patch.shape[0] - self.patch_size + 1, self.stride):
                for j in range(0, pre_patch.shape[1] - self.patch_size + 1, self.stride):
                    patch = {
                        'preimage': pre_patch[i:i+self.patch_size, j:j+self.patch_size],
                        'postimage': post_patch[i:i+self.patch_size, j:j+self.patch_size],
                        'premask': pre_mask_patch[i:i+self.patch_size, j:j+self.patch_size],
                        'postmask': post_mask_patch[i:i+self.patch_size, j:j+self.patch_size]
                    }
                    patches.append(patch)
        
        # 训练时随机选择patch，测试时顺序返回
        if self.quarter_idx < 0:
            patch = random.choice(patches)
        else:
            patch = patches[idx % len(patches)]
            
        # 应用数据增强
        if self.transform:
            transformed = self.transform(
                image=patch['postimage'],
                mask=patch['postmask'],
                image2=patch['preimage'],
                mask2=patch['premask']
            )
            patch['postimage'] = transformed['image']
            patch['postmask'] = transformed['mask']
            patch['preimage'] = transformed['image2']
            patch['premask'] = transformed['mask2']
        
        return {
            'preimage': torch.from_numpy(patch['preimage']).permute(2, 0, 1).float() / 255.0,
            'postimage': torch.from_numpy(patch['postimage']).permute(2, 0, 1).float() / 255.0,
            'premask': torch.from_numpy(patch['premask']).long(),
            'postmask': torch.from_numpy(patch['postmask']).long()
        }

    def __len__(self):
        return len(self.image_pairs)