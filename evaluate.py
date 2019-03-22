# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:11:34 2019

@author: zhxsking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import os
import shutil
import sys
import time
import copy
from boxx import show

from unet import UNet
from jsj_dataset import JsjDataset
from option import Option


def diceCoff(input, target):
    """计算dice系数"""
    eps = 1
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    return ((2 * inter.float() + eps) / union.float()).item()

def evalNet(net, loss_func, dataloader, device):
    """用验证集评判网络性能"""
    net.eval()
    dice_coff = 0
    loss_temp = 0
    with torch.no_grad():
        for cnt, (img, mask) in enumerate(dataloader, 1):
            img = img.to(device)
            mask = mask.to(device)
            out = net(img)
            loss = loss_func(out, mask)
            out_prob = torch.sigmoid(out)
            loss_temp += loss.item()
            dice_coff += diceCoff(out_prob, mask)
    return loss_temp / cnt, dice_coff / cnt


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    # 加载数据
#    dataset_test = JsjDataset(opt.dir_img_test, opt.dir_mask_test)
    dataset_test = JsjDataset(opt.dir_img_val, opt.dir_mask_val)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=512,
                            shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    state = torch.load(r"data\{}\final-unet.pkl".format(opt.name), map_location=opt.device)
    unet.load_state_dict(state['unet'])
    loss_func = nn.BCEWithLogitsLoss().to(opt.device)
    
#    loss_val, dice_val = evalNet(unet, loss_func, dataloader_test, opt.device)
    
    
    unet.eval()
    dice_coff = 0
    loss_temp = 0
    with torch.no_grad():
        for cnt, (img, mask) in enumerate(dataloader_test, 1):
            img = img.to(opt.device)
            mask = mask.to(opt.device)
            out = unet(img)
            loss = loss_func(out, mask)
            out_prob = torch.sigmoid(out)
            loss_temp += loss.item()
            dice_coff += diceCoff(out_prob, mask)
    loss_temp /= cnt
    dice_coff /= cnt
    
    print(loss_temp, dice_coff)

