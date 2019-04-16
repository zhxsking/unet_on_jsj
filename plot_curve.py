# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:47:43 2019

@author: zhxsking
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from option import Option


opt = Option()
state1 = torch.load(r"data\{}\final-unet.pkl".format(opt.name))
state2 = torch.load(r"data\{}\best-unet.pkl".format(opt.name))

loss_list_train = state1['loss_list_train']
loss_list_val = state1['loss_list_val']
dice_list_train = state1['dice_list_train']
dice_list_val = state1['dice_list_val']

best_dice = state2['best_dice']
best_epoch = state2['best_epoch']

#%% 画训练模型loss及dice图
fig = plt.figure(figsize=(20,8))

plt.subplot(121)
plt.plot(np.arange(1,101), loss_list_train)
plt.plot(np.arange(1,101), loss_list_val)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('{} Model Loss'.format(opt.name), fontsize=20)
plt.xlabel('epoch', fontsize=17)
plt.ylabel('loss', fontsize=17)
plt.legend(['train','validation'], fontsize=17, loc='upper right')

plt.subplot(122)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(np.arange(1,101), dice_list_train)
plt.plot(np.arange(1,101), dice_list_val)
plt.scatter(best_epoch, best_dice)
text_xy = {'RGB': (-50, -50), 'RGN': (-120, -50)}
plt.annotate('best dice: {:.4f}'.format(best_dice), xy=(best_epoch, best_dice-0.002),
             xycoords='data', xytext=text_xy[opt.name], textcoords='offset points',
             fontsize=17, arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.1"))

plt.title('{} Model Dice Coefficient'.format(opt.name), fontsize=20)
plt.xlabel('epoch', fontsize=17)
plt.ylabel('dice coefficient', fontsize=17)
plt.legend(['train','validation'], fontsize=17, loc='lower right')
plt.show()

plt.savefig(r'data/{}-loss-dice-20190320.jpg'.format(opt.name), dpi=300)

#%% 画不同输入情况模型性能图
loss_rgb = [0.0961, 0.1144, 0.1047]
loss_rgn = [0.1188, 0.1534, 0.1207]
dice_rgb = [0.9442, 0.9396, 0.9348]
dice_rgn = [0.9284, 0.9158, 0.9270]

x_data = np.arange(3)
total_width, n = 0.5, 2
width = total_width / n
x_data = x_data - (total_width - width) / 2

fig = plt.figure(figsize=(30,12))

# loss
plt.subplot(121)
plt.bar(x_data, loss_rgb,  width=width, label='RGB')
plt.bar(x_data + width, loss_rgn, width=width, label='Multispectral')
plt.ylabel('loss', fontsize=27)
plt.xticks(np.arange(3), ['original image input', '3 index input', '6 index input'], fontsize=20)
plt.yticks(np.linspace(0, 0.18, 4), fontsize=20)
plt.legend(fontsize=20, loc='upper right')

# dice
plt.subplot(122)
plt.bar(x_data, dice_rgb,  width=width, label='RGB')
plt.bar(x_data + width, dice_rgn, width=width, label='Multispectral')
plt.ylabel('dice', fontsize=27)
plt.ylim(0.8, 1)
plt.xticks(np.arange(3), ['original image input', '3 index input', '6 index input'], fontsize=20)
plt.yticks(np.linspace(0.8, 1, 5), fontsize=20)
plt.legend(fontsize=20, loc='upper right')

plt.show()
plt.savefig(r'data/loss-dice-change-bar.jpg', dpi=300)


