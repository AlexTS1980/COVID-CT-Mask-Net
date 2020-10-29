import os
import re
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn.functional as F
from torch.utils import data as data
from torchvision import transforms as transforms
import matplotlib.pyplot as plt
from skimage.measure import label as method_label
from skimage.measure import regionprops


# dataset for GGO and C segmentation
class CovidCTData(data.Dataset):

    def __init__(self, **kwargs):
        self.mask_type = kwargs['mask_type']
        self.ignore_ = kwargs['ignore_small']
        # ignore small areas?
        if self.ignore_:
           self.area_th = 100
        else:
           self.area_th = 1
        self.stage = kwargs['stage']
        # this returns the path to imgs dir
        self.data = kwargs['data']
        # this returns the path to
        self.gt = kwargs['gt']
        # IMPORTANT: the order of images and masks must be the same
        self.sorted_data = sorted(os.listdir(self.data))
        self.sorted_gt = sorted(os.listdir(self.gt))
        self.fname = None
        self.img_fname = None

    # this method normalizes the image and converts it to Pytorch tensor
    # Here we use pytorch transforms functionality, and Compose them together,
    def transform_img(self, img):
        # Faster R-CNN does the normalization
        t_ = transforms.Compose([
            # transforms.ToPILImage(),
            # transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        img = t_(img)
        return img

    # inputs: box coords (min_row, min_col, max_row, max_col)
    # array HxW from whic to extract a single object's mask
    # each isolated mask should have a different label, lab>0
    # masks are binary uint8 type
    def extract_single_mask(self, mask, lab):
        _mask = np.zeros(mask.shape, dtype=np.uint8)
        area = mask == lab
        _mask[area] = 1
        return _mask

    def load_img(self, idx):
        im = PILImage.open(os.path.join(self.data, self.sorted_data[idx]))
        self.img_fname = os.path.join(self.data, self.sorted_data[idx])
        im = self.transform_img(im)
        return im

    def load_labels_covid_ctscan_data(self, idx):
        list_of_bboxes = []
        labels = []
        list_of_masks = []
        # load bbox
        self.fname = os.path.join(self.gt, self.sorted_gt[idx])
        # extract bboxes from the mask
        mask = np.array(PILImage.open(self.fname))
        # only GGO: merge C and background
        # or merge GGO and C into a single mask
        # or keep separate masks
        if self.mask_type == "ggo":
           mask[mask==3] = 0
        elif self.mask_type == "merge":
           mask[mask==3] = 2
        # array  (NUM_CLASS_IN_IMNG, H,W) without bgr+lungs class (merge Class 0 and 1)
        # THIS IS IMPORTANT! CAN TRIGGER CUDA ERROR
        mask_classes = mask == np.unique(mask)[:, None, None][2:]
        # extract bounding boxes and masks for each object
        for _idx, m in enumerate(mask_classes):
            lab_mask = method_label(m)
            regions = regionprops(lab_mask)
            for _i, r in enumerate(regions):
                # get rid of really small ones:
                if r.area > self.area_th:
                    box_coords = (r.bbox[1], r.bbox[0], r.bbox[3], r.bbox[2])
                    list_of_bboxes.append(box_coords)
                    labels.append(_idx + 1)
                    # create a mask for one object, append to the list of masks
                    mask_obj = self.extract_single_mask(lab_mask, r.label)
                    list_of_masks.append(mask_obj)
        # create labels for Mask R-CNN
        # DO NOT CHANGE THESE DATATYPES!
        lab = {}
        list_of_bboxes = torch.as_tensor(list_of_bboxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.int64)
        masks = torch.tensor(list_of_masks, dtype=torch.uint8)
        lab['labels'] = labels
        lab['boxes'] = list_of_bboxes
        lab['masks'] = masks
        lab['fname'] = self.fname
        lab['img_name'] = self.img_fname
        return lab

    # 'magic' method: size of the dataset
    def __len__(self):
        return len(os.listdir(self.data))

    # return one datapoint
    def __getitem__(self, idx):
        X = self.load_img(idx)
        y = self.load_labels_covid_ctscan_data(idx)
        return X, y
