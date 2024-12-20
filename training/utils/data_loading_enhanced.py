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
    #                 patch_size=1024,
    #                 stride=60,
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
        
        # 打印目录内容
        print(f"Pre-image directory contents: {listdir(pre_dir_img)}")
        print(f"Post-image directory contents: {listdir(post_dir_img)}")
        
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
            
        # 如果指定quarter_idx，进行LOQO划分
        if quarter_idx is not None:
            self.quarter_idx = quarter_idx
            self.quarters_to_use = self._get_quarters_to_use(quarter_idx)
            
        # 只存储patches的索引信息
        self.patches = []
        for img_idx, (pre_file, post_file) in enumerate(self.image_pairs):
            # 获取图像大小而不加载整个图像
            with Image.open(Path(pre_dir_img) / pre_file) as img:
                h, w = img.size[1], img.size[0]
            
            mid_h, mid_w = h//2, w//2
            quarters = {
                0: (slice(0, mid_h), slice(0, mid_w)),
                1: (slice(0, mid_h), slice(mid_w, w)),
                2: (slice(mid_h, h), slice(0, mid_w)),
                3: (slice(mid_h, h), slice(mid_w, w))
            }
            
            # 为每个quarter生成patches的索引
            for q in self.quarters_to_use:
                h_slice, w_slice = quarters[q]
                quarter_h = h_slice.stop - h_slice.start
                quarter_w = w_slice.stop - w_slice.start
                
                for i in range(0, quarter_h - self.patch_size + 1, self.stride):
                    for j in range(0, quarter_w - self.patch_size + 1, self.stride):
                        self.patches.append({
                            'img_idx': img_idx,
                            'quarter': q,
                            'x': j,
                            'y': i
                        })
                        
        print(f"Total patches to be generated: {len(self.patches)}")

    def _get_quarters_to_use(self, quarter_idx):
        """确定使用哪些quarters
        quarter_idx < 0: 训练集(使用3个quarters)
        quarter_idx >= 0: 测试集(使用1个quarter)
        """
        if quarter_idx < 0:  
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
        # 获取patch的索引信息
        patch_info = self.patches[idx]
        img_idx = patch_info['img_idx']
        quarter = patch_info['quarter']
        x, y = patch_info['x'], patch_info['y']
        
        # 获取文件名
        pre_name, post_name = self.image_pairs[img_idx]
        
        # 只加载需要的quarter部分
        with Image.open(Path(self.pre_dir_img) / pre_name) as img:
            h, w = img.size[1], img.size[0]
        
        # # 对掩码进行膨胀处理
        # pre_mask = self._dilate_mask(pre_mask)
        # post_mask = self._dilate_mask(post_mask)
        
        # # 获取quarter的切片
        # h, w = pre_img.shape[:2]
        mid_h, mid_w = h//2, w//2
        quarters = {
            0: (slice(0, mid_h), slice(0, mid_w)),
            1: (slice(0, mid_h), slice(mid_w, w)),
            2: (slice(mid_h, h), slice(0, mid_w)),
            3: (slice(mid_h, h), slice(mid_w, w))
        }
        h_slice, w_slice = quarters[quarter]
        
        # 只读取需要的区域
        pre_img = np.array(Image.open(Path(self.pre_dir_img) / pre_name).crop((
            w_slice.start, h_slice.start, w_slice.stop, h_slice.stop
        )).convert('RGB'))
        post_img = np.array(Image.open(Path(self.post_dir_img) / post_name).crop((
            w_slice.start, h_slice.start, w_slice.stop, h_slice.stop
        )).convert('RGB'))
        pre_mask = np.array(Image.open(Path(self.pre_dir_mask) / pre_name).crop((
            w_slice.start, h_slice.start, w_slice.stop, h_slice.stop
        )).convert('L'))
        post_mask = np.array(Image.open(Path(self.post_dir_mask) / post_name).crop((
            w_slice.start, h_slice.start, w_slice.stop, h_slice.stop
        )).convert('L'))
        
        # 提取patch
        patch = {
            'preimage': pre_img[y:y+self.patch_size, x:x+self.patch_size],
            'postimage': post_img[y:y+self.patch_size, x:x+self.patch_size],
            'premask': pre_mask[y:y+self.patch_size, x:x+self.patch_size],
            'postmask': post_mask[y:y+self.patch_size, x:x+self.patch_size]
        }
        
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
        return len(self.patches)