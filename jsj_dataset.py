# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 16:36:40 2018

@author: zhxsking
"""

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from os import listdir
from os.path import join
from PIL import Image

from calc_vi import rgb2vis, rgn2vis

class JsjDataset(Dataset):
    def __init__(self, dir_img, dir_mask, do_vi=False):
        super().__init__()
        self.dir_img = dir_img
        self.dir_mask = dir_mask
        self.do_vi = do_vi
        
    def __getitem__(self, index):
        img_names = listdir(self.dir_img)
        mask_names = listdir(self.dir_mask)
        
        img = Image.open(join(self.dir_img, img_names[index]))
        mask = Image.open(join(self.dir_mask, mask_names[index]))
        
        if 'RGB' in self.dir_img:
            means = (0.57633764, 0.47007486, 0.3075999)
            stds =(0.2519291, 0.21737799, 0.17447254)
        elif 'RGN' in self.dir_img:
            means = (0.19842228, 0.15358844, 0.2672494)
            stds =(0.102274425, 0.07998896, 0.124288246)
        
        # 标准化
        img_process = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(means, stds),
                ])
        img = img_process(img)
        
        # 计算植被指数
        if self.do_vi:
            if 'RGB' in self.dir_img:
                # ['G-R','ExG','ExG2','MExG','ExR','ExR2','VDVI','NGBDI',
                # 'NGRDI','RGRI','GRRI','GBRI','BRRI','RGBVI','ExGR',
                # 'ExGR2','CIVE','CIVE2','VEG','COM','COM2']
                vi_types = ['ExG','ExR','VDVI','NGRDI','RGRI','ExGR']
                img = rgb2vis(img, vi_types)
            elif 'RGN' in self.dir_img:
                vi_types = ['NDVI','RVI','NDWI','DVI','PVI','SAVI']
                img = rgn2vis(img, vi_types)

        mask = transforms.ToTensor()(mask)
        
        return img, mask
    
    def __len__(self):
        return len(listdir(self.dir_img))
    
if __name__ == '__main__':
    __spec__ = None
    
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

#    torch.manual_seed(1)
    
    dir_img = r"E:\pic\jiansanjiang\contrast\RGB\data\train\img"
    dir_mask = r"E:\pic\jiansanjiang\contrast\RGB\data\train\mask"
    dataset = JsjDataset(dir_img, dir_mask, do_vi=True)
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=2)
    
    dataset_iter = iter(dataset)
    img_o, lab_o = dataset_iter.__next__()
    img = img_o.numpy()
    lab = lab_o.numpy()
    plt.figure(1)
    plt.subplot(221)
    plt.imshow(img[0], cmap='gray')
    plt.subplot(222)
    plt.imshow(lab[0], cmap='gray')
    
    dataloader_iter = iter(dataloader)
    img1_o, lab1_o = dataloader_iter.__next__()
    img1 = img1_o.numpy()
    lab1 = lab1_o.numpy()
    plt.subplot(223)
    plt.imshow(img1[0][0], cmap='gray')
    plt.subplot(224)
    plt.imshow(lab1[0][0], cmap='gray')
    
    for cnt, (img, mask) in enumerate(dataloader, 1):
        img_show = img.detach().numpy()[0][0]
        mask_show = mask.detach().numpy()[0][0]
        
        plt.figure()
        plt.subplot(121)
        plt.imshow(img_show, cmap='gray')
        plt.subplot(122)
        plt.imshow(mask_show, cmap='gray')
        plt.show()
        break
