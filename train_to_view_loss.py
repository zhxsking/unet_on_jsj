# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 10:03:56 2018
@author: zhxsking
可视化loss变化，用于调参
单独窗口显示 %matplotlib qt5
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from unet import UNet
from jsj_dataset import JsjDataset
from option import Option


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()
    
    dataset = JsjDataset(opt.dir_img_train, opt.dir_mask_train)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize,
                            shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_depth=opt.depth).to(opt.device)
    loss_func = nn.BCEWithLogitsLoss().to(opt.device)
    optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=10, gamma=0.5)
    
    # 加载预训练的参数
    if opt.pretrained:
        state = torch.load(opt.net_path)
        unet.load_state_dict(state['unet'])
        optimizer.load_state_dict(state['optimizer'])
    unet.train()
    
    loss_list = []
    plt.ion()
    plt.show()
    print('start...')
    for epoch in range(opt.epochs):
        loss_temp = 0
        scheduler.step()
        for cnt, (img, mask) in enumerate(dataloader, 1):
            img = img.to(opt.device)
            mask = mask.to(opt.device)
            out = unet(img)
            out_prob = torch.sigmoid(out)
            loss = loss_func(out, mask)
            
#                local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
#                print('epoch {}/{}, iter {}/{}, loss {}, {}'.format(epoch+1, opt.epochs, cnt, len(dataloader), loss, local_time))
            loss_temp += loss.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_temp /= cnt
        loss_list.append(loss_temp)
        
        # 绘制loss曲线
        plt.plot(loss_list)
        plt.title('loss {}'.format(loss_temp))
        plt.show()
        plt.pause(0.001)
        
        print('epoch {}/{} done, train loss {:.4f}'
              .format(epoch+1, opt.epochs, loss_temp))
    plt.ioff()
        

            
            
            