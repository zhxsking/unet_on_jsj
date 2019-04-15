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
        self.do_vi = True # 是否计算植被指数
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
        self.pretrained = False
        self.pretrained_net_path = r"data\{}\best-unet.pkl".format(self.name)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 预测相关参数
        self.block_size = 1280 # 一次处理block_size*block_size个像素大小的块
        self.use_dialog = False # 是否弹出对话框选择图片
        self.img_path = r"E:\pic\jiansanjiang\contrast\{}\img\{}.jpg".format(self.name, self.name)
        self.mask_path = r"E:\pic\jiansanjiang\contrast\{}\mask\{}.jpg".format(self.name, self.name)
        self.threshold = 0.5 # 阈值
        self.do_damage_eval = False # 是否进行灾损评估
        
        if 'RGB' in self.name:
            self.means = (0.57633764, 0.47007486, 0.3075999)
            self.stds =(0.2519291, 0.21737799, 0.17447254)
        elif 'RGN' in self.name:
            self.means = (0.19842228, 0.15358844, 0.2672494)
            self.stds =(0.102274425, 0.07998896, 0.124288246)
        if self.do_vi:
            if 'RGB' in self.name and self.depth == 3:
                self.means = (0.033005234, 0.33679792, 0.023215197)
                self.stds =(0.08267227, 0.15183108, 0.06558669)
            elif 'RGN' in self.name and self.depth == 3:
                self.means = (0.16403621, 1.4104799, -0.28379896)
                self.stds =(0.0715629, 0.21261951, 0.07500779)
            if 'RGB' in self.name and self.depth == 6:
                self.means = (0.033005234, 0.33679792, 0.023215197, -0.10389829, 1.2706941, -0.3037929)
                self.stds =(0.08267227, 0.15183108, 0.06558669, 0.09945874, 0.5774954, 0.15403529)
            elif 'RGN' in self.name and self.depth == 6:
                self.means = (0.16403621, 1.4104799, -0.28379896, 0.06882706, -0.7989329, 0.10449199)
                self.stds =(0.0715629, 0.21261951, 0.07500779, 0.037735604, 0.090508476, 0.044216525)

