# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 19:34:46 2019

@author: zhxsking
"""

import torch

class Option():
    """定义网络的参数及其他"""
    def __init__(self):
        # 训练相关参数
        self.name = 'RGB'
        self.depth = 3 # 图片深度
        self.epochs = 100
        self.batchsize = 12 # 3 12
        self.lr = 5e-4
        self.weight_decay = 0.000
        self.workers = 0 # 多进程，可能会卡程序
        self.dir_img_train = r"E:\pic\jiansanjiang\contrast\{}\data\train\img".format(self.name) # 训练集
        self.dir_mask_train = r"E:\pic\jiansanjiang\contrast\{}\data\train\mask".format(self.name)
        self.dir_img_val = r"E:\pic\jiansanjiang\contrast\{}\data\val\img".format(self.name) # 验证集
        self.dir_mask_val = r"E:\pic\jiansanjiang\contrast\{}\data\val\mask".format(self.name)
        self.dir_img_test = r"E:\pic\jiansanjiang\contrast\{}\data\test\img".format(self.name) # 验证集
        self.dir_mask_test = r"E:\pic\jiansanjiang\contrast\{}\data\test\mask".format(self.name)
        self.pretrained = True
        self.pretrained_net_path = r"data\{}\best-unet.pkl".format(self.name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 预测相关参数
        self.block_size = 1280 # 一次处理block_size*block_size个像素大小的块
        self.use_dialog = False # 是否弹出对话框选择图片
        self.img_path = r"E:\pic\jiansanjiang\contrast\{}\img\{}.jpg".format(self.name, self.name)
        self.mask_path = r"E:\pic\jiansanjiang\contrast\{}\mask\{}.jpg".format(self.name, self.name)
        self.threshold = 0.5 # 阈值
        self.do_damage_eval = False # 是否进行灾损评估
        
