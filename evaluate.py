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
from tqdm import tqdm
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


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    # 加载数据
    dataset_test = JsjDataset(opt.dir_img_test, opt.dir_mask_test)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    state = torch.load(r"data\{}\best-unet.pkl".format(opt.name), map_location=opt.device)
    unet.load_state_dict(state['unet'])
    loss_func = nn.BCEWithLogitsLoss().to(opt.device)
    
    # 验证
    unet.eval()
    dice_coff = 0
    loss_temp = 0
    pbar = tqdm(total=len(dataloader_test), desc='Evaluating')
    with torch.no_grad():
        for cnt, (img, mask) in enumerate(dataloader_test, 1):
            pbar.update(1)
            img = img.to(opt.device)
            mask = mask.to(opt.device)
            out = unet(img)
            loss = loss_func(out, mask)
            out_prob = torch.sigmoid(out)
            loss_temp += loss.item()
            dice_coff += diceCoff(out_prob, mask)
    pbar.close()
    loss_temp /= cnt
    dice_coff /= cnt
    
    print('loss {:.4f}, dice {:.4f}'.format(loss_temp, dice_coff))

