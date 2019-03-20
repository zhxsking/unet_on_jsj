# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:03:56 2018
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
    
    # 初始化保存目录
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    else:
        shutil.rmtree('checkpoint')
        os.makedirs('checkpoint')
    
    # 加载数据
    dataset = JsjDataset(opt.dir_img_train, opt.dir_mask_train)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers)   
    dataset_val = JsjDataset(opt.dir_img_val, opt.dir_mask_val)
    dataloader_val = DataLoader(dataset=dataset_val, batch_size=opt.batchsize,
                                shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    loss_func = nn.BCEWithLogitsLoss().to(opt.device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10, gamma=0.5) # 动态改变lr
    
    # 加载预训练的参数
    if opt.pretrained:
        state = torch.load(opt.pretrained_net_path)
        unet.load_state_dict(state['unet'])
        optimizer.load_state_dict(state['optimizer'])
    
    # 开始训练
    since = time.time()# 记录时间
    loss_list_train = []
    dice_list_train = []
    loss_list_val = []
    dice_list_val = []
    best_dice = 0.0
    best_epoch = 1
    best_model = copy.deepcopy(unet.state_dict())
    print('start...')
    for epoch in range(opt.epochs):
        loss_temp = 0.0
        dice_temp = 0.0
        scheduler.step()
        unet.train()
        for cnt, (img, mask) in enumerate(dataloader, 1):
            img = img.to(opt.device)
            mask = mask.to(opt.device)
            out = unet(img)
            loss = loss_func(out, mask)
            
#            local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#            print('epoch {}/{}, iter {}/{}, loss {}, {}'.format(epoch+1,
#                  opt.epochs, cnt, len(dataloader), loss, local_time))
            
            loss_temp += loss.item()
            out_prob = torch.sigmoid(out)
            dice_temp += diceCoff(out_prob, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_temp /= cnt
        dice_temp /= cnt
        loss_list_train.append(loss_temp)
        dice_list_train.append(dice_temp)
        
        # 验证
        loss_val, dice_val = evalNet(unet, loss_func, dataloader_val, opt.device)
        loss_list_val.append(loss_val)
        dice_list_val.append(dice_val)
        
        # 更新最优模型
        if dice_val >= best_dice:
            best_epoch = epoch + 1
            best_dice = dice_val
            best_model = copy.deepcopy(unet.state_dict())
            
        print('''epoch {}/{} done, train loss {:.4f}, train dice {:.4f}, val loss {:.4f}, val dice {:.4f}'''
          .format(epoch+1, opt.epochs, loss_temp, dice_temp, loss_val, dice_val))
        
        # 保存中途模型
        torch.save(unet.state_dict(), r'checkpoint/unet-epoch-{}.pkl'.format(epoch+1))
        
    # 保存最佳模型
    best_unet_state = {
            'best_epoch': best_epoch,
            'best_dice': best_dice,
            'unet': best_model,
            }
    torch.save(best_unet_state, r'checkpoint/best-unet.pkl')
    
    # 保存最终模型以及参数
    time_elapsed = time.time() - since # 用时
    final_unet_state = {
            'epoch': epoch+1,
            'time': time_elapsed,
            'loss_list_train': loss_list_train,
            'dice_list_train': dice_list_train,
            'loss_list_val': loss_list_val,
            'dice_list_val': dice_list_val,
            'optimizer': optimizer.state_dict(),
            'unet': unet.state_dict(),
            }
    torch.save(final_unet_state, r'checkpoint/final-unet.pkl')
    
    # 统计用时并显示训练信息
    
    print('-' * 50)
    print('Training complete in {:.0f}m {:.0f}s'
          .format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Dice {:4f} in epoch {}'.format(best_dice, best_epoch))
    
    # 训练完显示loss及dice曲线
    plt.figure()
    plt.subplot(121)
    plt.plot(loss_list_train)
    plt.plot(loss_list_val)
    plt.subplot(122)
    plt.plot(dice_list_train)
    plt.plot(dice_list_val)
    plt.show()
        

            
            
            