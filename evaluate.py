# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:11:34 2019

@author: zhxsking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from boxx import show

from unet import UNet
from jsj_dataset import JsjDataset
from option import Option
from jsj_utils import Record


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
    dataset_test = JsjDataset(opt.dir_img_test, opt.dir_mask_test, do_vi=opt.do_vi)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    state = torch.load(r"data\{}\best-unet.pkl".format(opt.name), map_location=opt.device)
    unet.load_state_dict(state['unet'])
    loss_func = nn.BCEWithLogitsLoss().to(opt.device)
    
    if 'RGB' in opt.name:
        means = (0.57633764, 0.47007486, 0.3075999)
        stds =(0.2519291, 0.21737799, 0.17447254)
    elif 'RGN' in opt.name:
        means = (0.19842228, 0.15358844, 0.2672494)
        stds =(0.102274425, 0.07998896, 0.124288246)
    
    # 验证
    unet.eval()
    dice_coff = Record()
    loss_temp = Record()
    pbar = tqdm(total=len(dataloader_test), desc='Evaluating')
    with torch.no_grad():
        for cnt, (img, mask) in enumerate(dataloader_test, 1):
            pbar.update(1)
            img = img.to(opt.device)
            mask = mask.to(opt.device)
            out = unet(img)
            loss = loss_func(out, mask)
            out_prob = torch.sigmoid(out)
            
            loss_temp.update(loss.item(), img.shape[0])
            dice_coff.update(diceCoff(out_prob, mask), img.shape[0])
            
            # 保存一部分结果
            if not(opt.do_vi):
                if not os.path.exists(r'data\{}\res'.format(opt.name)):
                    os.makedirs(r'data\{}\res'.format(opt.name))
                if cnt in [10, 20, 30]:
                    # 取3张图片
                    for j in range(3):
                        tmp_img = img.cpu().numpy().transpose(0,2,3,1)[j,:]
                        tmp_out = out_prob.cpu().numpy().transpose(0,2,3,1)[j,:,:,0]
                        
                        # 反归一化
                        for d in range(3):
                            tmp_img[:,:,d] = tmp_img[:,:,d] * stds[d] + means[d]
                        plt.imsave(r'data\{}\res\{}-{}-img.jpg'.format(opt.name, cnt, j), tmp_img)
                        plt.imsave(r'data\{}\res\{}-{}-out.jpg'.format(opt.name, cnt, j), tmp_out, cmap='gray')

    pbar.close()
    
    print('{} loss {:.4f}, dice {:.4f}'.format(opt.name, loss_temp.avg, dice_coff.avg))

