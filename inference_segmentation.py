
# Mask R-CNN model for lesion segmentation in chest CT scans
# Torchvision detection package is locally re-implemented
# by Alex Ter-Sarkisov@City, University of London
# alex.ter-sarkisov@city.ac.uk
# 2020
import argparse
import time
import pickle
import copy
import torch
import torchvision
import numpy as np
import os
import cv2
import models.mask_net as mask_net
from models.mask_net.rpn_segmentation import AnchorGenerator
from models.mask_net.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from models.mask_net.covid_mask_net import MaskRCNNHeads, MaskRCNNPredictor
from torchvision import transforms
from torch.utils import data
import torch.utils as utils
import datasets.dataset_segmentation as dataset
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import utils
import config_segmentation as config


# main method
def main(config, step):
    devices = ['cpu', 'cuda']
    mask_classes = ['both', 'ggo', 'merge']
    backbones = ['resnet50', 'resnet34', 'resnet18']
    truncation_levels = ['0','1','2']
    assert config.device in devices
    assert config.backbone_name in backbones
    assert config.truncation in truncation_levels

    assert config.mask_type in mask_classes
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # get the configuration
    # get the thresholds
    confidence_threshold, mask_threshold, save_dir, data_dir, img_dir, gt_dir, mask_type, rpn_nms, roi_nms, backbone_name, truncation \
        = config.confidence_th, config.mask_logits_th, config.save_dir, config.test_data_dir, config.test_imgs_dir, \
        config.gt_dir, config.mask_type, config.rpn_nms_th, config.roi_nms_th, config.backbone_name, config.truncation

    if mask_type == "both":
        n_c = 3
    else:
        n_c = 2
    ckpt = torch.load(config.ckpt, map_location=device)

    model_name = None
    if 'model_name' in ckpt.keys():
        model_name = ckpt['model_name']
    sizes = ckpt['anchor_generator'].sizes
    aspect_ratios = ckpt['anchor_generator'].aspect_ratios
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    print("Anchors: ", anchor_generator.sizes, anchor_generator.aspect_ratios)

    # create modules
    # this assumes FPN with 256 channels
    box_head = TwoMLPHead(in_channels=7 * 7 * 256, representation_size=128)
    if backbone_name == 'resnet50':
       maskrcnn_heads = None
       box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
       mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=n_c)
    else:
       #Backbone->FPN->boxhead->boxpredictor
       box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
       maskrcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
       mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=n_c)

    # keyword arguments
    maskrcnn_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 100,
                     'box_nms_thresh': roi_nms, 'box_score_thresh': confidence_threshold, 'rpn_nms_thresh': rpn_nms,
                     'box_head': box_head, 'rpn_anchor_generator': anchor_generator, 'mask_head':maskrcnn_heads,
                     'mask_predictor': mask_predictor, 'box_predictor': box_predictor}

    # Instantiate the segmentation model
    maskrcnn_model = mask_net.maskrcnn_resnet_fpn(backbone_name, truncation, pretrained_backbone=False, **maskrcnn_args)
    # Load weights
    maskrcnn_model.load_state_dict(ckpt['model_weights'])
    # Set to evaluation mode
    print(maskrcnn_model)
    maskrcnn_model.eval().to(device)

    start_time = time.time()
    # get the correct masks and mask colors
    if mask_type == "ggo":
       ct_classes = {0: '__bgr', 1: 'GGO'}
       ct_colors = {1: 'red', 'mask_cols': np.array([[255, 0, 0]])}
    elif mask_type == "merge":
       ct_classes = {0: '__bgr', 1: 'Lesion'}
       ct_colors = {1: 'red', 'mask_cols': np.array([[255, 0, 0]])}
    elif mask_type == "both":
       ct_classes = {0: '__bgr', 1: 'GGO', 2: 'CL'}
       ct_colors = {1: 'red', 2: 'blue', 'mask_cols': np.array([[255, 0, 0], [0, 0, 255]])} 

    if not save_dir in os.listdir('.'):
       os.mkdir(save_dir)

    # model name from config, not checkpoint
    if model_name is None:
        model_name = "maskrcnn_segmentation"
    elif model_name is not None and config.model_name != model_name:
        print("Using model name from the config.")
        model_name = config.model_name

    # run the inference with provided hyperparameters
    test_ims = os.listdir(os.path.join(data_dir, img_dir))
    for j, ims in enumerate(test_ims):
        step(os.path.join(os.path.join(data_dir, img_dir), ims), device, maskrcnn_model, model_name,
             confidence_threshold, mask_threshold, save_dir, ct_classes, ct_colors, j)
    end_time = time.time()
    print("Inference took {0:.1f} seconds".format(end_time - start_time))


def test_step(image, device, model, model_name, theta_conf, theta_mask, save_dir, cls, cols, num):
    im = PILImage.open(image)
    # convert image to RGB, remove the alpha channel
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    img = np.array(im)
    # copy image to make background for plotting
    bgr_img = copy.deepcopy(img)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    # torchvision transforms, the rest Mask R-CNN does internally
    t_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()])
    img = t_(img).to(device)
    out = model([img])
    # scores + bounding boxes + labels + masks
    scores = out[0]['scores']
    bboxes = out[0]['boxes']
    classes = out[0]['labels']
    mask = out[0]['masks']
    # this is the array for all masks
    best_scores = scores[scores > theta_conf]
    # Are there any detections with confidence above the threshold?
    if len(best_scores):
        best_idx = np.where(scores > theta_conf)
        best_bboxes = bboxes[best_idx]
        best_classes = classes[best_idx]
        best_masks = mask[best_idx]
        print('bm', best_masks.shape)
        mask_array = np.zeros([best_masks[0].shape[1], best_masks[0].shape[2], 3], dtype=np.uint8)
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(12, 6)
        ax.axis("off")
        # plot predictions
        for idx, dets in enumerate(best_bboxes):
            found_masks = best_masks[idx][0].detach().clone().to(device).numpy()
            pred_class = best_classes[idx].item()
            pred_col_n = cols[pred_class]
            pred_class_txt = cls[pred_class]
            pred_col = cols['mask_cols'][pred_class - 1]
            mask_array[found_masks > theta_mask] = pred_col
            rect = Rectangle((dets[0], dets[1]), dets[2] - dets[0], dets[3] - dets[1], linewidth=1,
                             edgecolor=pred_col_n, facecolor='none', linestyle="--")
            ax.text(dets[0] + 40, dets[1], '{0:}'.format(pred_class_txt), fontsize=10, color=pred_col_n)
            ax.text(dets[0], dets[1], '{0:.2f}'.format(best_scores[idx]), fontsize=10, color=pred_col_n)
            ax.add_patch(rect)

        added_image = cv2.addWeighted(bgr_img, 0.5, mask_array, 0.75, gamma=0)
        ax.imshow(added_image)
        fig.savefig(os.path.join(save_dir, model_name + "_" + str(num) + ".png"),
                    bbox_inches='tight', pad_inches=0.0)

    else:
        print("No detections")

# run the inference
if __name__ == '__main__':
    config_test = config.get_config_pars("test")
    main(config_test, test_step)
