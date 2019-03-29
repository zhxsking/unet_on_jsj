# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:38:34 2019
画多边形并抠出相应的mask，左键选点，右键停止
@author: zhxsking
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

from option import Option


def on_mouse(event, x, y, flags, param):
    img, mask, pts = param
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print (x,y)
        cv2.circle(img, (x,y), 10, (255,255,255), 10)
        pts.append((x, y))
        if len(pts) >= 2:
            cv2.line(img, pts[-2], pts[-1], (0,255,255), 6)
        cv2.imshow('image', img)
        
    if event == cv2.EVENT_RBUTTONDOWN:
        if len(pts) >= 2:
            cv2.line(img, pts[0], pts[-1], (0,255,255), 6)
            cv2.imshow('image', img)
        
        pts = np.array(pts)
        
#        mask_tmp = cv2.polylines(mask, [pts], True, (255))
#        mask_tmp = cv2.fillPoly(mask_tmp, [pts], (255))
#        cv2.imshow('mask', mask_tmp)
        
        mask = cv2.polylines(mask, [pts], True, (1))
        mask = cv2.fillPoly(mask, [pts], (1))

def damage_eval(opt, res):
    img = cv2.imread(opt.img_path)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    pts = []
    
    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', on_mouse, [img, mask, pts])
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    
    rgb = plt.imread(opt.img_path)
    mask_tmp = mask.copy().astype(np.float)
    mask_tmp[mask==1] = 0.7
    mask_tmp[mask==0] = 1
    res_rgb = rgb * np.stack((mask_tmp, mask_tmp, mask_tmp), axis=2) # 二值结果映射到原图
    res_rgb = res_rgb.astype(np.uint8)
    
    mask_res = mask * res
    damage_ratio = mask_res.sum() / mask.sum()
    
    # 保存结果
    res_path = r'data\{}\res-block-{:.4f}.jpg'.format(opt.name, damage_ratio)
    plt.imsave(res_path, res_rgb)
    
    return damage_ratio


if __name__ == '__main__':
    opt = Option()
    img = cv2.imread(opt.img_path)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    pts = []
    
    cv2.namedWindow('image', 0)
    cv2.setMouseCallback('image', on_mouse, [img, mask, pts])
    
    cv2.imshow('image',img)
    cv2.waitKey(0)
    
    rgb = plt.imread(opt.img_path)
    mask_tmp = mask.copy().astype(np.float)
    mask_tmp[mask==1] = 0.7
    mask_tmp[mask==0] = 1
    res_rgb = rgb * np.stack((mask_tmp, mask_tmp, mask_tmp), axis=2) # 二值结果映射到原图
    res_rgb = res_rgb.astype(np.uint8)
    
    plt.imshow(res_rgb)
