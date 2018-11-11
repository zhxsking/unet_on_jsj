# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:03:56 2018
@author: zhxsking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from unet import UNet
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
from PIL import Image
import numpy as np


class Option():
    """超参数定义类"""
    def __init__(self):
        self.in_dim = 3 # 图片按rgb输入还是按灰度输入，可选1,3
        self.scale = 0.5 # 图片缩放
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            torch.backends.cudnn.benchmark = True
        self.net_path = r"checkpoint\unet-epoch26.pkl"
        self.img_path = r"D:\pic\carvana\just_for_test\train\0cdf5b5d0ce1_02.jpg"
        self.use_dialog = True


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    # 选择一副图片
    if opt.use_dialog:
        root = tk.Tk()
        img_path = filedialog.askopenfilename(initialdir=opt.img_path.split('train')[0])
        root.withdraw()
        if not(img_path): sys.exit(0)
    else:
        img_path = opt.img_path
    # 读取图片并裁剪成左右两部分  
    read_mode = 'L' if opt.in_dim==1 else ('RGB' if opt.in_dim==3 else 'error')
    img_ori = Image.open(img_path).convert(read_mode)
    height_ori = img_ori.height
    width_ori = img_ori.width
    img = img_ori.resize(tuple(map(lambda x: int(x * opt.scale), img_ori.size)))
    img_left = img.crop((0, 0, img.height, img.height)) # 图片的左半边，640*640，参数为左上右下
    img_right = img.crop((img.width-img.height, 0, img.width, img.height)) # 图片的右半边，640*640
    img_left = transforms.functional.to_tensor(img_left).unsqueeze(0)
    img_right = transforms.functional.to_tensor(img_right).unsqueeze(0)
    # 加载模型
    unet = UNet(in_dim=opt.in_dim)
    if opt.cuda:
        unet = unet.cuda()
        img_left = img_left.cuda()
        img_right = img_right.cuda()
        state = torch.load(opt.net_path)
    else:
        state = torch.load(opt.net_path, map_location='cpu')
    unet.load_state_dict(state['net'])
    print('load model done, calculate...')
    unet.eval()
    # 预测
    with torch.no_grad():
        out_left = unet(img_left)
        out_right = unet(img_right)
        out_left_prob = F.sigmoid(out_left).squeeze(0)
        out_right_prob = F.sigmoid(out_right).squeeze(0)
        # 转换为PIL并上采样
        post_proc = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(height_ori),
                transforms.ToTensor(),
                ])
        # 转换回numpy类型
        out_left_np = post_proc(out_left_prob.cpu()).squeeze(0).numpy()
        out_right_np = post_proc(out_right_prob.cpu()).squeeze(0).numpy()
        # 左右两部分合并
        out = np.zeros((height_ori, width_ori), np.float32)
        out[:, :width_ori//2] = out_left_np[:, :width_ori//2]
        out[:, width_ori//2:] = out_right_np[:, -(width_ori//2+1):-1]
        output = (out > 0.5).astype(np.float32) # 二值化
        output_rgb = img_ori * np.stack((output,output,output), axis=2)
        output_rgb = output_rgb.astype(np.uint8)

        plt.figure()
        plt.subplot(131)
        plt.imshow(img_ori)
        plt.subplot(132)
        plt.imshow(output, cmap='gray')
        plt.subplot(133)
        plt.imshow(output_rgb)
        plt.show()

            
            