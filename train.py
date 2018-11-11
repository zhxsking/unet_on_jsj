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
from jsj_dataset import JsjDataset
from unet import UNet
import matplotlib.pyplot as plt
from os.path import join
import sys

class Option():
    """超参数定义类"""
    def __init__(self):
        self.epochs = 50
        self.batchsize = 1
        self.lr = 1e-3
        self.in_dim = 3 # 图片按rgb输入还是按灰度输入，可选1,3
        self.scale = 0.5 # 图片缩放
        self.workers = 2 # 多进程读取data
        self.dir_img = r"D:\pic\jiansanjiang\data\img"
        self.dir_mask = r"D:\pic\jiansanjiang\data\mask"
        self.save_path = r"checkpoint"
        self.cuda = False
        if torch.cuda.is_available():
            self.cuda = True
            torch.backends.cudnn.benchmark = True
        self.pretrained = False
        self.net_path = r"checkpoint\unet-epoch26.pkl"


if __name__ == '__main__':
    __spec__ = None
    
    opt = Option()
    
    dataset = JsjDataset(opt.dir_img, opt.dir_mask, scale=opt.scale)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batchsize, shuffle=True, num_workers=opt.workers)
    
    unet = UNet(in_dim=opt.in_dim)
    loss_func = nn.BCEWithLogitsLoss()
    if opt.cuda:
        unet = unet.cuda()
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(unet.parameters(), lr=opt.lr, weight_decay=0.0005)
    # 加载预训练的参数
    if opt.pretrained:
        state = torch.load(opt.net_path)
        unet.load_state_dict(state['net'])
        optimizer.load_state_dict(state['optimizer'])
    unet.train()
    
    loss_list = []
    loss_list_big = []
    try:
        for epoch in range(opt.epochs):
            print('epoch {}/{} start...'.format(epoch+1, opt.epochs))
            loss_temp = 0
            
            for cnt, (img, mask) in enumerate(dataloader, 1):
                if opt.cuda:
                    img = img.cuda()
                    mask = mask.cuda()
                    
                out = unet(img)
                out_prob = F.sigmoid(out)
                
                if opt.cuda:
                    out_show = out.detach().cpu().numpy()[0][0]
                    out_prob_show = out_prob.detach().cpu().numpy()[0][0]
                else:
                    out_show = out.detach().numpy()[0][0]
                    out_prob_show = out_prob.detach().numpy()[0][0]
                
                plt.figure()
                plt.subplot(121)
                plt.imshow(out_show, cmap='gray')
                plt.subplot(122)
                plt.imshow(out_prob_show, cmap='gray')
                plt.show()
                torchvision.utils.save_image(out_prob, join(opt.save_path, r'output\epoch{}-iter{}.jpg'.format(epoch+1, cnt)))
                
                loss = loss_func(out, mask)
                print('epoch {}, iter {}, loss {}'.format(epoch+1, cnt, loss))
                loss_temp += loss.item()
                loss_list_big.append(loss.item())
    
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            loss_temp /= cnt
            loss_list.append(loss_temp)
            print('epoch {} done, average loss {}'.format(epoch+1, loss_temp))
            
            # 保存模型
            if (epoch+1) % 1 == 0:
                state = {
                        'epoch': epoch+1,
                        'loss_list': loss_list,
                        'optimizer': optimizer.state_dict(),
                        'net': unet.state_dict(),
                    }
                torch.save(state, join(opt.save_path, 'unet-epoch{}.pkl'.format(epoch+1)))
    except KeyboardInterrupt:
        print('Interrupt!')
        plt.figure()
        plt.subplot(121)
        plt.plot(loss_list)
        plt.subplot(122)
        plt.plot(loss_list_big)
        plt.show()
        state = {
                'epoch': epoch+1,
                'loss_list': loss_list,
                'optimizer': optimizer.state_dict(),
                'net': unet.state_dict(),
            }
        torch.save(state, join(opt.save_path, 'unet-epoch{}-iter{}.pkl'.format(epoch+1,cnt)))
        sys.exit(0)
        

            
            
            