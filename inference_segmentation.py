# COVID Mask R-CNN project
# You must have Torchvision v0.3.0+
#
import argparse
import time
import pickle
import copy
import torch
import torchvision
import numpy as np
import os
import cv2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
from torchvision import transforms
from torch.utils import data
import torch.utils as utils
import datasets.dataset_segmentation as dataset
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import utils
import config


# main method
def main(config, step):
    devices = ['cpu', 'cuda']
    ct_classes = {0: '__bgr', 1: 'GGO', 2: 'CL'}
    ct_colors = {1: 'red', 2: 'blue', 'mask_cols': np.array([[255, 0, 0], [0, 0, 255]])}
    
    assert config.device in devices
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #
    ckpt = torch.load(config.ckpt, map_location=device)
    ###############################################################################################
    # MASK R-CNN model
    # Alex: these settings could also be added to the config
    maskrcnn_args = {'min_size': 512, 'max_size': 1024, 'box_nms_thresh': 0.75, 'rpn_nms_thresh': 0.75, 'num_classes':None}
    # Alex: for Ground glass opacity and consolidatin segmentation
    anchor_generator = ckpt['anchor_generator']
    print("Anchors: ", anchor_generator)
    # num_classes:3 (1+2)
    box_head_input_size = 256 * 7 * 7
    box_head = TwoMLPHead(in_channels=box_head_input_size, representation_size=128)
    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=3)
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3], output_size=14, sampling_ratio=2)
    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=3)

    maskrcnn_args['rpn_anchor_generator'] = anchor_generator
    maskrcnn_args['mask_roi_pool'] = mask_roi_pool
    maskrcnn_args['mask_predictor'] = mask_predictor
    maskrcnn_args['box_predictor'] = box_predictor
    maskrcnn_args['box_head'] = box_head
    # Instantiate the segmentation model
    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False,
                                                                        progress=True, **maskrcnn_args)
    # Load weights
    maskrcnn_model.load_state_dict(ckpt['model_weights'])
    # Set to evaluation mode
    print(maskrcnn_model)
    maskrcnn_model.eval().to(device)

    start_time = time.time()
    # get the thresholds
    confidence_threshold, mask_threshold, save_dir, data_dir, imgs_dir, model_name = \
        config.confidence_th, config.mask_logits_th, config.save_dir, config.test_data_dir, config.test_imgs_dir, \
        ckpt['model_name']
 
    if not save_dir in os.listdir('.'):
       os.mkdir(save_dir)
    if model_name is None:
       model_name = "maskrcnn_segmentation"
    # run the inference with provided hyperparameters
    test_ims = os.listdir(os.path.join(data_dir, imgs_dir))
    for j, ims in enumerate(test_ims):
        step(os.path.join(os.path.join(data_dir, imgs_dir), ims), device, maskrcnn_model, model_name,
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
