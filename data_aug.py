# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:36:57 2018
@author: zhxsking

数据增广，并将增广后的数据分两个文件夹保存
"""

import Augmentor
import os, shutil
import sys
import tkinter as tk
from tkinter import filedialog

IMG_NUM = 5000 # 输出图片数量

if __name__ == '__main__':
    # 打开image文件夹对话框
    root = tk.Tk()
    img_dir = filedialog.askdirectory(title='选择图片所在文件夹')
    if not(img_dir): 
        root.withdraw()
        sys.exit(0)
    # 打开mask文件夹对话框
    mask_dir = filedialog.askdirectory(title='选择mask所在文件夹')
    root.withdraw()
    if not(mask_dir): sys.exit(0)
    
    p = Augmentor.Pipeline(img_dir)
    p.ground_truth(mask_dir)
    # 增强操作
    p.crop_by_size(1, width=1280, height=1280, centre=False)
    p.flip_left_right(0.5)
    p.flip_top_bottom(0.5)
#    p.random_erasing(0.5, rectangle_area=0.5) # 随机遮挡
    p.rotate(0.5, max_left_rotation=10, max_right_rotation=10)
    p.rotate_random_90(0.5) # 随机旋转90、180、270度，注意图片需为方的
    p.zoom_random(0.3, percentage_area=0.5) # 随机放大
    p.random_distortion(0.3,grid_height=5,grid_width=5,magnitude=5) # 弹性扭曲
    p.shear(0.3, max_shear_left=5, max_shear_right=5) # 随机错切（斜向一边）
    p.skew(0.3, magnitude=0.3) # 透视形变
    p.sample(IMG_NUM, multi_threaded=True) # 多线程提速但占内存，输出大图慎用多线程防死机
    
    # 原始输出图片保存路径
    out_dir = os.path.join(img_dir, 'output')
    # 分两个文件夹重新保存image和mask
    save_dir_img = os.path.join(out_dir, 'img')
    save_dir_mask = os.path.join(out_dir, 'mask')
    if not os.path.exists(save_dir_img):
        os.mkdir(save_dir_img)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)
    # 移动到新文件夹并改名
    out_names = os.listdir(out_dir)
    for out_name in out_names:
        if 'original' in out_name:
            new_name = out_name.split('jpg_')[1]
            shutil.move(os.path.join(out_dir, out_name), os.path.join(save_dir_img, new_name))
        elif 'groundtruth' in out_name:
            new_name = out_name.split('jpg_')[1]
            shutil.move(os.path.join(out_dir, out_name), os.path.join(save_dir_mask, new_name))
    
    
    
    
    
    
    
    
    