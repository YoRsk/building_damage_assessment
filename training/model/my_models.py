from .my_functions import *

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision.models import vgg13_bn, vgg16_bn, vgg16, vgg19, vgg19_bn
resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# resnet = torchvision.models.resnet.resnet50(pretrained=True)

class SiameseUNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=5):
        super().__init__()
        # resnet = ResNet50WithCBAM(use_cbam=True, image_depth=3, num_classes=n_classes)
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
class SiamUNetConCVgg19(nn.Module):
    """Unet with VGG-13 (with BN), VGG-16 (with BN) encoder.
    """

    def __init__(self,out_channels=5):
        super().__init__()
        self.bn = False
        self.up_sample_mode = 'conv_transpose'
        self.encoder = vgg19(pretrained=True).features
        self.block1 = nn.Sequential(*self.encoder[:4])
        self.block2 = nn.Sequential(*self.encoder[4:9])
        self.block3 = nn.Sequential(*self.encoder[9:18])
        self.block4 = nn.Sequential(*self.encoder[18:27])
        self.block5 = nn.Sequential(*self.encoder[27:36])

        self.bottleneck = nn.Sequential(*self.encoder[36:])
        # Upsampling Path
        self.up_conv5 = UpBlockMid(512,1536,512, self.up_sample_mode,self.bn)
        self.up_conv4 = UpBlockMid(512,1536,256,self.up_sample_mode,self.bn)
        self.up_conv3 = UpBlockMid(256,768,128,self.up_sample_mode,self.bn)
        self.up_conv2 = UpBlockMid(128,384,64,self.up_sample_mode,self.bn)
        self.up_conv1 = UpBlockMid(64,192,64,self.up_sample_mode,self.bn)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x1,x2):
        #input A
        x11 = self.block1(x1)
        x12 = self.block2(x11)
        x13 = self.block3(x12)
        x14 = self.block4(x13)
        x15 = self.block5(x14)
        #input B
        x21 = self.block1(x2)
        x22 = self.block2(x21)
        x23 = self.block3(x22)
        x24 = self.block4(x23)
        x25 = self.block5(x24)
        
        
        x = self.bottleneck(x25)
        x = self.up_conv5(x, torch.cat((x15,x25),dim = 1))
        x = self.up_conv4(x, torch.cat((x14,x24),dim = 1))
        x = self.up_conv3(x, torch.cat((x13,x23),dim = 1))
        x = self.up_conv2(x, torch.cat((x12,x22),dim = 1))
        x = self.up_conv1(x, torch.cat((x11,x21),dim = 1))
        x = self.conv_last(x)
        return x