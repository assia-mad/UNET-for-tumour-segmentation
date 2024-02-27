import torch.nn as nn
import torchvision.transforms.functional as tf
import torch

class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1)
            self.batchnorm = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU()
        def forward(self, x):
            x = self.conv(x)
            x = self.batchnorm(x)
            x = self.relu(x)
            return x

class StackEncoder(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(StackEncoder, self).__init__()
            self.max_pool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
            self.block = nn.Sequential(
                ConvBlock(in_channels, out_channels),
                ConvBlock(out_channels, out_channels))
        def forward(self, x):
            block_out = self.block(x)
            pool_out = self.max_pool(block_out)
            return block_out, pool_out
        
class StackDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
            super(StackDecoder, self).__init__()
            self.block = nn.Sequential(
                ConvBlock(in_channels+in_channels, out_channels),
                ConvBlock(out_channels, out_channels))
    def forward(self, x, concat_tensor):
        batch, channels, height, width = concat_tensor.shape
        x = torch.nn.functional.interpolate(x, size=(height, width))
        x = torch.cat([x, concat_tensor], 1)
        blockout = self.block(x)
        return blockout


class UNET(nn.Module):
    def __init__(self, input_shape):
        super(UNET, self).__init__()
        self.batch, self.channel, self.height, self.width = input_shape
        self.down1 = StackEncoder(self.channel, 64)
        self.down2 = StackEncoder(64, 128)
        self.down3 = StackEncoder(128, 256)
        self.bottleneck = ConvBlock(256, 256)
        self.up3 = StackDecoder(256, 128)
        self.up2 = StackDecoder(128, 64)
        self.up1 = StackDecoder(64, 1) # 96 x 96 x 1
    def forward(self, x):
        down1, out = self.down1(x)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        bottleneck = self.bottleneck(out)
        up3 = self.up3(x=bottleneck, concat_tensor=down3)
        up2 = self.up2(x=up3, concat_tensor=down2)
        up1 = self.up1(x=up2, concat_tensor=down1)
        return up1