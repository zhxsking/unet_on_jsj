# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 19:38:16 2019

@author: zhxsking
"""

import numpy as np
import matplotlib.pyplot as plt


# 计算图片的均值及方差
img = plt.imread(r'E:\pic\jiansanjiang\contrast\RGB\img\rgb.jpg')
img = img.astype(np.float32) / 255
means, stds = [], []
for i in range(3):
    means.append(img[:,:,i].mean())
    stds.append(img[:,:,i].std())
print('rgb mean std:')
print(means, stds)

img = plt.imread(r'E:\pic\jiansanjiang\contrast\RGN\img\rgn.jpg')
img = img.astype(np.float32) / 255
means, stds = [], []
for i in range(3):
    means.append(img[:,:,i].mean())
    stds.append(img[:,:,i].std())
print('rgn mean std:')
print(means, stds)
