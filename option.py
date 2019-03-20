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
        self.name = 'RGN'
        self.depth = 3 # 图片深度
        self.epochs = 20
        self.batchsize = 3 # 3 12
        self.lr = 5e-4
        self.weight_decay = 0.000
        self.workers = 0 # 多进程，可能会卡程序
        self.dir_img_train = r"E:\pic\jiansanjiang\contrast\{}\data\train\img".format(self.name) # 训练集
        self.dir_mask_train = r"E:\pic\jiansanjiang\contrast\{}\data\train\mask".format(self.name)
        self.dir_img_val = r"E:\pic\jiansanjiang\contrast\{}\data\val\img".format(self.name) # 验证集
        self.dir_mask_val = r"E:\pic\jiansanjiang\contrast\{}\data\val\mask".format(self.name)
        self.pretrained = False
        self.pretrained_net_path = r"checkpoint\best-unet.pkl"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 预测相关参数
        self.block_size = 640 # 一次处理block_size*block_size个像素大小的块
        self.use_dialog = False # 是否弹出对话框选择图片
        self.img_path = r"E:\pic\jiansanjiang\img\10.jpg"
        self.mask_path = r"E:\pic\jiansanjiang\mask\10.jpg"
        self.threshold = -0.7 # 阈值,ln(0.5)=-0.69
        
