# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 15:32:08 2019

@author: zhxsking
"""
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from boxx import show


def rgb2vi(img, vi_type):
    '''
    RGB图像转换为植被指数
    'G-R','ExG','ExG2','MExG','ExR','ExR2','VDVI','NGBDI','NGRDI','RGRI','GRRI',
    'GBRI','BRRI','RGBVI','ExGR','ExGR2','NHLVI','CIVE','CIVE2','VEG','COM','COM2'
    '''
    r, g, b = img[:,:]
    
    if vi_type == 'G-R':
        vi = g - r
    elif vi_type == 'ExG':
        vi = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
    elif vi_type == 'ExG2':
        vi = 2 * g - r - b
    elif vi_type == 'MExG':
        vi = 1.262 * (g / (r+g+b)) - 0.884 * (r / (r+g+b)) - 0.311 * (b / (r+g+b))
    elif vi_type == 'ExR':
        vi = 1.4 * r - g
    elif vi_type == 'ExR2':
        vi = 1.4 * (r / (r+g+b)) - (g / (r+g+b))
    elif vi_type == 'VDVI':
        vi = (2 * g - r - b) / (2 * g + r + b)
    elif vi_type == 'NGBDI':
        vi = (g - b) / (g + b)
    elif vi_type == 'NGRDI':
        vi = (g - r) / (g + r)
    elif vi_type == 'RGRI':
        vi = r / g
    elif vi_type == 'GRRI':
        vi = g / r
    elif vi_type == 'GBRI':
        vi = g / b
    elif vi_type == 'BRRI':
        vi = b / r
    elif vi_type == 'RGBVI':
        vi = (g*g - r*b) / (g*g + r*b)
    elif vi_type == 'ExGR':
        vi = (2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))) - (1.4 * (r) - (g))
    elif vi_type == 'ExGR2':
        vi = (2 * g - r - b) - (1.4 * r - g)
#    elif vi_type == 'NHLVI':
#        hsl = colorspace('HSL<-RGB', img)
#        vi = (hsl(:,:,1) - hsl(:,:,3)) ./ (hsl(:,:,1) + hsl(:,:,3))
    elif vi_type == 'CIVE':
        vi = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
    elif vi_type == 'CIVE2':
        vi = 0.441*r - 0.881*g + 0.385*b + 18.78745
    elif vi_type == 'VEG':
        vi = g / (r**0.667 * b**(1 - 0.667))
    elif vi_type == 'COM':
        vi1 = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
        vi2 = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
        vi3 = (2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))) - (1.4 * (r / (r+g+b)) - (g / (r+g+b)))
        vi4 = g / (r**0.667 * b**(1 - 0.667))
        vi = vi1 + vi2 + vi3 + vi4
    elif vi_type == 'COM2':
        vi1 = 2 * (g / (r+g+b)) - (r / (r+g+b)) - (b / (r+g+b))
        vi2 = 0.441*(r / (r+g+b)) - 0.881*(g / (r+g+b)) + 0.385*(b / (r+g+b)) + 18.78745
        vi3 = g / (r**0.667 * b**(1 - 0.667))
        vi = 0.36 * vi1 + 0.47 * vi2 + 0.17 * vi3
    else:
        raise Exception('Unknown VI')
    return vi

def rgb2vis(img, vi_types):
    vis = []
    for vi_type in vi_types:
        vi = rgb2vi(img, vi_type).numpy()
        vis.append(vi)
    vis = np.array(vis)
    vis = torch.Tensor(vis)
    return vis

if __name__ == '__main__':
    path = r'E:\pic\jiansanjiang\contrast\RGB\data\test\img\0a54a1b8-b743-4824-8b5e-8e64893b7d64.jpg'
    img = Image.open(path)
    
    means = (0.57633764, 0.47007486, 0.3075999)
    stds =(0.2519291, 0.21737799, 0.17447254)
    
    
    img_process = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(means, stds),
            ])
    
    img = img_process(img)
    
    vi = rgb2vi(img, 'ExG')
    vi_types = ['G-R','ExG','ExG2','MExG','ExR','ExR2','VDVI','NGBDI','NGRDI',
                'RGRI','GRRI','GBRI','BRRI','RGBVI','ExGR','ExGR2',
                'CIVE','CIVE2','VEG','COM','COM2']
    vis = rgb2vis(img, vi_types)

    show(vis)






