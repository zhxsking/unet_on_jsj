# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 10:36:57 2018
@author: zhxsking

数据增广，并将增广后的数据按比例分训练集、验证集、测试集保存
"""

import Augmentor
import os, shutil
import sys
import random
import tkinter as tk
from tkinter import filedialog

IMG_NUM = 5000 # 输出图片数量
ratio_train = 0.7 # 训练集比例
ratio_validation = 0.15 # 验证集比例
ratio_test = 0.15 # 测试集比例

def list_split(full_list, ratio_1=0.7, ratio_2=0.15, ratio_3=0.15, shuffle=True):
    """按比例切分list"""
    # 先乘以100防止float数据丢失信息
    ratio_1 = int(ratio_1 * 100)
    ratio_2 = int(ratio_2 * 100)
    ratio_3 = int(ratio_3 * 100)
    if ((ratio_1 + ratio_2 + ratio_3) > 100):
        raise Exception("比例和超过1!")
    n_total = len(full_list)
    offset_1 = int(n_total * ratio_1 / 100)
    offset_2 = int(n_total * (ratio_1+ratio_2) / 100)
    offset_3 = int(n_total * (ratio_1+ratio_2+ratio_3) / 100)
    if n_total==0 or offset_1<1:
        return full_list, [], []
    if shuffle:
        random.shuffle(full_list)
    sublist_1 = full_list[:offset_1]
    sublist_2 = full_list[offset_1:offset_2]
    sublist_3 = full_list[offset_2:offset_3]
    return sublist_1, sublist_2, sublist_3

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
    # 切分数据集
    out_names_full = os.listdir(save_dir_img)
    out_names_train, out_names_validation, out_names_test = list_split(out_names_full,
                                                                       ratio_1 = ratio_train,
                                                                       ratio_2 = ratio_validation,
                                                                       ratio_3 = ratio_test)
    # 建立训练集、验证集、测试集文件夹
    save_dir_train = os.path.join(out_dir, 'train')
    save_dir_validation = os.path.join(out_dir, 'validation')
    save_dir_test = os.path.join(out_dir, 'test')
    if not os.path.exists(save_dir_train): os.mkdir(save_dir_train)
    if not os.path.exists(save_dir_validation): os.mkdir(save_dir_validation)
    if not os.path.exists(save_dir_test): os.mkdir(save_dir_test)
    # 将图片放入训练集文件夹
    train_dir_img = os.path.join(save_dir_train, 'img')
    train_dir_mask = os.path.join(save_dir_train, 'mask')
    if not os.path.exists(train_dir_img): os.mkdir(train_dir_img)
    if not os.path.exists(train_dir_mask): os.mkdir(train_dir_mask)
    for out_name in out_names_train:
        shutil.move(os.path.join(save_dir_img, out_name), os.path.join(train_dir_img, out_name))
        shutil.move(os.path.join(save_dir_mask, out_name), os.path.join(train_dir_mask, out_name))
    # 将图片放入验证集文件夹
    validation_dir_img = os.path.join(save_dir_validation, 'img')
    validation_dir_mask = os.path.join(save_dir_validation, 'mask')
    if not os.path.exists(validation_dir_img): os.mkdir(validation_dir_img)
    if not os.path.exists(validation_dir_mask): os.mkdir(validation_dir_mask)
    for out_name in out_names_validation:
        shutil.move(os.path.join(save_dir_img, out_name), os.path.join(validation_dir_img, out_name))
        shutil.move(os.path.join(save_dir_mask, out_name), os.path.join(validation_dir_mask, out_name))
    # 将图片放入测试集文件夹    
    test_dir_img = os.path.join(save_dir_test, 'img')
    test_dir_mask = os.path.join(save_dir_test, 'mask')
    if not os.path.exists(test_dir_img): os.mkdir(test_dir_img)
    if not os.path.exists(test_dir_mask): os.mkdir(test_dir_mask)
    for out_name in out_names_test:
        shutil.move(os.path.join(save_dir_img, out_name), os.path.join(test_dir_img, out_name))
        shutil.move(os.path.join(save_dir_mask, out_name), os.path.join(test_dir_mask, out_name))
    # 删除建立的两个文件夹
    os.rmdir(save_dir_img)
    os.rmdir(save_dir_mask)
    

    
    
    
    
    
    
    
    
    