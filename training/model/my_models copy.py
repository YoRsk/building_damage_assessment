from .my_functions import *

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# resnet = torchvision.models.resnet.resnet50(pretrained=True)

class SiameseUNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5):
        super().__init__()
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # resnet = torchvision.models.resnet.resnet50(pretrained=True)
        #weights=ResNet50_Weights.IMAGENET1K_V1
        down_blocks = []
        up_blocks = []

        # input_block 包含了ResNet的前三个层（通常是卷积层、BN层、ReLU层等）。这些层用于提取特征。
        # CNN卷积层的基本组成单元标配：Conv + BN +ReLU 三剑客
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        # input_pool 保存了第四个层，通常是最大池化层。
        #池化（Pooling）：也称为欠采样或下采样。主要用于特征降维，
        # 压缩数据和参数的数量，减小过拟合，同时提高模型的容错性
        self.input_pool = list(resnet.children())[3]

        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(1024, 1024)
        up_blocks.append(SiameseUp(2048, 6144, 1024,False))
        up_blocks.append(SiameseUp(1024, 512 + (1024*2), 512,False))
        up_blocks.append(SiameseUp(512, 256 + (512*2), 256,False))
        up_blocks.append(SiameseUp(256, 128 + (256*2), 128,False))
        up_blocks.append(SiameseUp(128, 64 + (64*2), 64,False))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = OutConv(64, n_classes)

    def forward(self, x1, x2):
        dblock = []
        for i, block in enumerate(self.down_blocks, 2):
            dblock.append(block)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
        x11 = self.input_block(x1)
        x21 = self.input_block(x2)
        xp12 = self.input_pool(x11)
        xp22 = self.input_pool(x21)
        x12 = dblock[0](xp12)
        x22 = dblock[0](xp22)
        x13 = dblock[1](x12)
        x23 = dblock[1](x22)
        x14 = dblock[2](x13)
        x24 = dblock[2](x23)
        #x15 = dblock[3](x14)
        #x25 = dblock[3](x24)
        xbridge = self.bridge(x24)
        #xu1 = self.up_blocks[0](xbridge,torch.cat((x15,x25),dim = 1))
        xu2 = self.up_blocks[1](xbridge,torch.cat((x14,x24),dim = 1))
        xu3 = self.up_blocks[2](xu2,torch.cat((x13,x23),dim = 1))
        xu4 = self.up_blocks[3](xu3,torch.cat((x12,x22),dim = 1))
        xu5 = self.up_blocks[4](xu4,torch.cat((x11,x21),dim = 1))
        x = self.out(xu5)
        return x
        '''pre_pools1 = dict()
        pre_pools2 = dict()
        pre_pools1[f"layer_0"] = x1
        pre_pools2[f"layer_0"] = x2
        x1 = self.input_block(x1)
        x2 = self.input_block(x2)
        pre_pools1[f"layer_1"] = x1
        pre_pools2[f"layer_1"] = x2
        x1 = self.input_pool(x1)
        x2 = self.input_pool(x2)

        for i, block in enumerate(self.down_blocks, 2):
            x1 = block(x1)
            x2 = block(x2)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x1
            pre_pools[f"layer_{i}"] = x2
        '''
