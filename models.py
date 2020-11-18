import torch
import torch.nn as nn

from .models_parts import *


class CNN_block(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(CNN_block, self).__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, [3, 3], 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.MaxPool2d([2, 2], 2)
        )

    def forward(self, x):

        return self.layer(x)


class Vanilla_CNN(nn.Module):

    def __init__(self, width):

        super(Vanilla_CNN, self).__init__()

        self.block1 = CNN_block(1, int(width * 0.5))
        self.block2 = CNN_block(int(width * 0.5), width)
        self.block3 = CNN_block(width, width)
        self.block4 = CNN_block(width, width)
        self.block5 = CNN_block(width, width)
        self.block6 = CNN_block(width, width)
        self.block7 = CNN_block(width, width)

    def forward(self, x):

        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = self.block6(out)
        out = self.block7(out)

        return out


class Simple_regressor(nn.Module):

    def __init__(self, in_channels, out_channels=1):

        super(Simple_regressor, self).__init__()

        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        return self.layer(x)

class UNet(nn.Module):

    def __init__(self, n_class):
        super().__init__()
                
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)
        
        self.conv_last = nn.Conv2d(64, n_class, 1)
        
        
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        return out