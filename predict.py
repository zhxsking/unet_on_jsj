# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:03:56 2018
@author: zhxsking
选择一张图像进行预测
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
from math import ceil


class Option():
    """超参数定义类"""
    def __init__(self):
        self.in_dim = 3 # 图片按rgb输入还是按灰度输入，可选1,3
        self.scale = 0.5 # 图片缩放
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            torch.backends.cudnn.benchmark = True
        self.net_path = r"checkpoint\unet-epoch20.pkl"
        self.img_path = r"E:\pic\jiansanjiang\data\img\00b9447f-314c-4ea8-8bf5-293e2f9b5356.jpg"
        self.use_dialog = True
        self.crop_width = 1280
        self.crop_height = 1280


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    # 加载模型
    unet = UNet(in_dim=opt.in_dim)
    if opt.cuda:
        unet = unet.cuda()
        state = torch.load(opt.net_path)
    else:
        state = torch.load(opt.net_path, map_location='cpu')
    unet.load_state_dict(state['net'])
    loss_list = state['loss_list']
    unet.eval()
    print('load model done!')
    # 选择一副图片
    if opt.use_dialog:
        root = tk.Tk()
        img_path = filedialog.askopenfilename(initialdir=opt.img_path.split('img')[0])
        root.withdraw()
        if not(img_path): sys.exit(0)
    else:
        img_path = opt.img_path
    # 读取图片并转换为numpy
    read_mode = 'L' if opt.in_dim==1 else ('RGB' if opt.in_dim==3 else 'error')
    img_ori = Image.open(img_path).convert(read_mode)
    height_ori = img_ori.height
    width_ori = img_ori.width
    img_np = np.array(img_ori)
    # 初始化预测图片
    res = np.zeros((height_ori, width_ori), np.float32)
    # 滑窗截图并预测
    dx, dy = opt.crop_width, opt.crop_height # 滑窗slide
    x_start, y_start, x_stop, y_stop = 0, 0, 0, 0 # 初始化索引起点与终点
    for x_ in range(ceil(width_ori / dx)):
        for y_ in range(ceil(height_ori / dy)):
            if (x_+1)*dx > width_ori and (y_+1)*dy <= height_ori: # 索引仅超出右边界则往左边多取一些
                y_start = y_ * dy
                y_stop = (y_ + 1) * dy
                x_start = width_ori - dx
                x_stop = width_ori
            elif (x_+1)*dx <= width_ori and (y_+1)*dy > height_ori: # 索引仅超出下边界则往上边多取一些
                y_start = height_ori - dy
                y_stop = height_ori
                x_start = x_ * dx
                x_stop = (x_ + 1) * dx
            elif (x_+1)*dx > width_ori and (y_+1)*dy > height_ori: # 索引超出右下边界则取右下角一块
                y_start = height_ori - dy
                y_stop = height_ori
                x_start = width_ori - dx
                x_stop = width_ori
                img_slice = img_np[-dy : -1, -dx : -1]
            elif (x_+1)*dx <= width_ori and (y_+1)*dy <= height_ori: # 一般情况
                y_start = y_ * dy
                y_stop = (y_ + 1) * dy
                x_start = x_ * dx
                x_stop = (x_ + 1) * dx
            # 取出一个窗口
            img_slice = img_np[y_start : y_stop, x_start : x_stop]
            # 输入unet的预处理
            img = Image.fromarray(img_slice)
            img = img.resize(tuple(map(lambda x: int(x * opt.scale), img.size)))
            img = transforms.functional.to_tensor(img).unsqueeze(0)
            if opt.cuda: img = img.cuda()
            # 预测
            with torch.no_grad():
                out = unet(img)
                out_prob = F.sigmoid(out).squeeze(0)
                # 转换为PIL并上采样
                post_proc = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(opt.crop_height),
                        transforms.ToTensor(),
                        ])
                # 转换回numpy类型
                out_np = post_proc(out_prob.cpu()).squeeze(0).numpy()
                # 将窗口的预测结果映射回大图
                res[y_start : y_stop, x_start : x_stop] = out_np
    res_bw = (res > 0.5).astype(np.float32) # 二值化
    res_rgb = img_ori * np.stack((res_bw, res_bw, res_bw), axis=2)
    res_rgb = res_rgb.astype(np.uint8)

    plt.figure()
    plt.plot(loss_list)

    plt.figure()
    plt.subplot(131)
    plt.imshow(img_ori)
    plt.subplot(132)
    plt.imshow(res_bw, cmap='gray')
    plt.subplot(133)
    plt.imshow(res_rgb)
    plt.show()
                


            
            