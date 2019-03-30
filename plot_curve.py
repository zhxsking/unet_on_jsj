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

#%% 
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
