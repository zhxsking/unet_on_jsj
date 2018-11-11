# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 17:30:06 2018
@author: zhxsking
"""

import torch
import torch.nn as nn

class Down(nn.Module):
    """unet的下降部分模块"""
    def __init__(self, in_channel, out_channel, do_pool=True):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                )
        self.pool = nn.MaxPool2d(2)
        self.do_pool = do_pool
        
    def forward(self, x):
        if self.do_pool:
            x = self.pool(x)
        x = self.conv(x)
        return x
        
    
class Up(nn.Module):
    """unet的上升部分模块"""
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                )
        
    def forward(self, x_prev, x):
        x = self.up(x)
        x = torch.cat((x_prev, x), dim=1)
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    """unet定义
    
    采用双线性上采样，实际网络与论文有所不同，
    论文中上卷积将图的深度也减半了，但上采样不改变深度
    """
    
    def __init__(self, in_dim):
        super().__init__()
        self.down1 = Down(in_dim, 64, do_pool=False)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.down5 = Down(512, 512)
        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)
        
    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x = self.up1(x4, x5)
        x = self.up2(x3, x)
        x = self.up3(x2, x)
        x = self.up4(x1, x)
        x = self.out_conv(x)
        return x
    

if __name__ == '__main__':
    unet = UNet(in_dim=1)
    if torch.cuda.is_available():
        unet = unet.cuda()
#    print(unet)

#    test_x = torch.FloatTensor(1, 1, 256, 256)
#    out_x = unet(test_x)
#    print(out_x.size())
    
    from torchsummary import summary
    summary(unet, (1,128,128))
        
        