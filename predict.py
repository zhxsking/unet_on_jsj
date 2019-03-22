# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:03:56 2018
@author: zhxsking
选择一张图像进行预测
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from PIL import Image
import numpy as np
from math import ceil

from unet import UNet
from jsj_dataset import JsjDataset
from option import Option


def diceCoff(input, target):
    """计算dice系数"""
    eps = 1.
    inter = np.dot(input.ravel(), target.ravel())
    union = np.sum(input) + np.sum(target) + eps
    return (2 * inter.astype(np.float32) + eps) / union.astype(np.float32)


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    state = torch.load(r"data\{}\best-unet.pkl".format(opt.name), map_location=opt.device)
    unet.load_state_dict(state['unet'])
    unet.eval()
    print('load model done!')
    
    # 选择图片
    if opt.use_dialog:
        # 打开待预测图片对话框
        root = tk.Tk()
        img_path = filedialog.askopenfilename(initialdir=opt.img_path.split('img')[0],
                                              title='选择待预测图片')
        if not(img_path):
            root.withdraw()
            sys.exit(0)
        
        # 打开mask图片对话框
        mask_path = filedialog.askopenfilename(initialdir=opt.mask_path.split('mask')[0],
                                           title='选择mask图片')
        root.withdraw()
        if not(mask_path): sys.exit(0) 
    else:
        img_path = opt.img_path
        mask_path = opt.mask_path
    
    # 读取图片并转换为numpy
    read_mode = 'L' if opt.depth==1 else ('RGB' if opt.depth==3 else 'error')
    img_ori = Image.open(img_path).convert(read_mode)
    height_ori = img_ori.height
    width_ori = img_ori.width
    img_np = np.array(img_ori)
    
    if 'RGB' in opt.name:
        means = (0.57633764, 0.47007486, 0.3075999)
        stds =(0.2519291, 0.21737799, 0.17447254)
    elif 'RGN' in opt.name:
        means = (0.19842228, 0.15358844, 0.2672494)
        stds =(0.102274425, 0.07998896, 0.124288246)
    
    # 初始化预测图片
    res = np.zeros((height_ori, width_ori), np.float32)
    
    # 滑窗截图并预测
    dx, dy = opt.block_size, opt.block_size # 滑窗slide
    x_start, y_start, x_stop, y_stop = 0, 0, 0, 0 # 初始化索引起点与终点
    block_num_h = int(ceil(height_ori / dy))
    block_num_w = int(ceil(width_ori / dx))
    
    pbar = tqdm(total=block_num_h*block_num_w, desc='Predicting')
    for x_ in range(block_num_w):
        for y_ in range(block_num_h):
            pbar.update(1)
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
            pre_proc = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(means, stds),
                    ])
            img = pre_proc(img_slice).unsqueeze(0).to(opt.device)
            
            # 预测
            with torch.no_grad():
                out = unet(img)
                out_prob = torch.sigmoid(out).squeeze(0)
    
                # 转换回numpy类型
                out_np = out_prob.cpu().squeeze(0).numpy()
                
                # 将窗口的预测结果映射回大图
                res[y_start : y_stop, x_start : x_stop] = out_np
    pbar.close()
    res_bw = (res > opt.threshold).astype(np.float32) # 二值化
    res_rgb = img_ori * np.stack((res_bw, res_bw, res_bw), axis=2) # 二值结果映射到原图
    res_rgb = res_rgb.astype(np.uint8)

    mask_ori = Image.open(mask_path).convert('L')
    mask_np = np.array(mask_ori)
    mask_bw = (mask_np > 128).astype(np.float32) # 二值化
    
    # 计算准确率
    dice_coff = diceCoff(res_bw, mask_bw)
    
    
    print('{} accuracy: {:.4f}'.format(opt.img_path, dice_coff))
    
    plt.figure()
    plt.subplot(131)
    plt.imshow(img_ori)
    plt.subplot(132)
    plt.imshow(res_bw, cmap='gray')
    plt.subplot(133)
    plt.imshow(res_rgb)
    plt.title('accuracy: {:.4f}'.format(dice_coff))
    plt.xticks([])
    plt.yticks([])
    plt.show()
 
            
            