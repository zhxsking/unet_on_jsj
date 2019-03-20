# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:47:43 2019

@author: zhxsking
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
from PIL import Image
import numpy as np
from math import ceil

from unet import UNet
from jsj_dataset import JsjDataset
from option import Option


opt = Option()
state = torch.load(r"data\final-unet.pkl", map_location=opt.device)

loss_list_train = state['loss_list_train']
loss_list_val = state['loss_list_val']
dice_list_train = state['dice_list_train']
dice_list_val = state['dice_list_val']

#%% 
fig = plt.figure(figsize=(20,8))

plt.subplot(121)
plt.plot(loss_list_train)
plt.plot(loss_list_val)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title('Model Loss', fontsize=20)
plt.xlabel('epoch', fontsize=17)
plt.ylabel('loss', fontsize=17)
plt.legend(['train','validation'], fontsize=17, loc='upper right')

plt.subplot(122)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.plot(dice_list_train)
plt.plot(dice_list_val)
plt.title('Model Dice Coefficient', fontsize=20)
plt.xlabel('epoch', fontsize=17)
plt.ylabel('dice coefficient', fontsize=17)
plt.legend(['train','validation'], fontsize=17, loc='lower right')
plt.show()

plt.savefig(r'data/rgb-loss-dice-20190320.jpg', dpi=300)
