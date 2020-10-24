import os,sys
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms

# dataset for sign detection and char detection
class COVID_CT_DATA(data.Dataset):

     def __init__(self, **kwargs):           
         super(COVID_CT_DATA).__init__()
         self.stage = kwargs['stage']
         # this returns the path to data dir
         self.data = kwargs['data']         
         self.fs = sorted(os.listdir(self.data))
         self.size = kwargs['img_size']
         # this returns the path to 
         self.img_fname = None

     def transform_img(self, img):
         # Faster R-CNN does the normalization
         t_ = transforms.Compose([
                             #transforms.ToPILImage(),
                             transforms.Resize(self.size),
                             transforms.ToTensor(),
                             ])
         img = t_(img)
         return img

     def load_img_label(self, idx):
         lab=torch.zeros(3, dtype=torch.float)
         lab[int(self.fs[idx].split('_')[0])] = 1
         im = PILImage.open(os.path.join(self.data, self.fs[idx]))
         if im.mode !='RGB':
            im = im.convert(mode='RGB')
         im = self.transform_img(im)
         return im, lab

     #'magic' method: size of the dataset
     def __len__(self):
         return len(os.listdir(self.data))

     # return one datapoint
     def __getitem__(self, idx):
         X,y = self.load_img_label(idx)
         return X,y


