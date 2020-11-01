# Mask R-CNN model for lesion segmentation in chest CT scans
# Torchvision detection package is locally re-implemented
# by Alex Ter-Sarkisov@City, University of London
# alex.ter-sarkisov@city.ac.uk
# 2020
import argparse
import time
import pickle
import torch
import torchvision
import numpy as np
import os, sys
import cv2
import models.mask_net as mask_net
from models.mask_net.rpn_segmentation import AnchorGenerator
from models.mask_net.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from models.mask_net.covid_mask_net import MaskRCNNHeads, MaskRCNNPredictor
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
def main(config, main_step):
    devices = ['cpu', 'cuda']
    mask_classes = ['both', 'ggo', 'merge']
    truncation_levels = ['0','1','2']
    backbones = ['resnet50', 'resnet34', 'resnet18']
    assert config.backbone_name in backbones
    assert config.mask_type in mask_classes
    assert config.truncation in truncation_levels

    # import arguments from the config file
    start_epoch, model_name, use_pretrained_resnet_backbone, num_epochs, save_dir, train_data_dir, val_data_dir, imgs_dir, gt_dir, batch_size, device, save_every, lrate, rpn_nms, mask_type, backbone_name, truncation = \
        config.start_epoch, config.model_name, config.use_pretrained_resnet_backbone, config.num_epochs, config.save_dir, \
        config.train_data_dir, config.val_data_dir, config.imgs_dir, config.gt_dir, config.batch_size, config.device, config.save_every, config.lrate, config.rpn_nms_th, config.mask_type, config.backbone_name, config.truncation

    assert device in devices
    if not save_dir in os.listdir('.'):
       os.mkdir(save_dir)

    if batch_size > 1:
        print("The model was implemented for batch size of one")
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)

    # Load the weights if provided
    if config.pretrained_model is not None:
       pretrained_model = torch.load(config.pretrained_model, map_location = device)
       use_pretrained_resnet_backbone = False
    else:
       pretrained_model=None
    torch.manual_seed(time.time())
    ##############################################################################################
    # DATASETS + DATALOADERS
    # Alex: could be added in the config file in the future
    # parameters for the dataset
    dataset_covid_pars_train = {'stage': 'train', 'gt': os.path.join(train_data_dir, gt_dir),
                                'data': os.path.join(train_data_dir, imgs_dir), 'mask_type':mask_type, 'ignore_small':True}
    datapoint_covid_train = dataset.CovidCTData(**dataset_covid_pars_train)

    dataset_covid_pars_eval = {'stage': 'eval', 'gt': os.path.join(val_data_dir, gt_dir),
                               'data': os.path.join(val_data_dir, imgs_dir), 'mask_type':mask_type, 'ignore_small':True}
    datapoint_covid_eval = dataset.CovidCTData(**dataset_covid_pars_eval)
    ###############################################################################################
    dataloader_covid_pars_train = {'shuffle': True, 'batch_size': batch_size}
    dataloader_covid_train = data.DataLoader(datapoint_covid_train, **dataloader_covid_pars_train)
    #
    dataloader_covid_pars_eval = {'shuffle': False, 'batch_size': batch_size}
    dataloader_covid_eval = data.DataLoader(datapoint_covid_eval, **dataloader_covid_pars_eval)
    ###############################################################################################
    # MASK R-CNN model
    # Alex: these settings could also be added to the config
    if mask_type == "both":
        n_c = 3
    else:
        n_c = 2
    maskrcnn_args = {'min_size': 512, 'max_size': 1024, 'rpn_batch_size_per_image': 256, 'rpn_positive_fraction': 0.75,
                     'box_positive_fraction': 0.75, 'box_fg_iou_thresh': 0.75, 'box_bg_iou_thresh': 0.5,
                     'num_classes': None, 'box_batch_size_per_image': 256, 'rpn_nms_thresh': rpn_nms}

    # Alex: for Ground glass opacity and consolidatin segmentation
    # many small anchors
    # use all outputs of FPN
    # IMPORTANT!! For the pretrained weights, this determines the size of the anchor layer in RPN!!!!
    # pretrained model must have anchors
    if pretrained_model is None:
       anchor_generator = AnchorGenerator(
           sizes=tuple([(2, 4, 8, 16, 32) for r in range(5)]),
           aspect_ratios=tuple([(0.1, 0.25, 0.5, 1, 1.5, 2) for rh in range(5)]))
    else:
       print("Loading the anchor generator")
       sizes = pretrained_model['anchor_generator'].sizes
       aspect_ratios = pretrained_model['anchor_generator'].aspect_ratios
       anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=aspect_ratios)
       print(anchor_generator, anchor_generator.num_anchors_per_location())
    # num_classes:3 (1+2)
    # in_channels
    # 256: number if channels from FPN
    # For the ResNet50+FPN: keep the torchvision architecture, but with 128 features
    # For lightweights models: re-implement MaskRCNNHeads with a single layer
    box_head = TwoMLPHead(in_channels=256*7*7,representation_size=128)
    if backbone_name == 'resnet50':
       maskrcnn_heads = None
       box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
       mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=n_c)
    else:
       #Backbone->FPN->boxhead->boxpredictor
       box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
       maskrcnn_heads = MaskRCNNHeads(in_channels=256, layers=(128,), dilation=1)
       mask_predictor = MaskRCNNPredictor(in_channels=128, dim_reduced=128, num_classes=n_c)

    maskrcnn_args['box_head'] = box_head
    maskrcnn_args['rpn_anchor_generator'] = anchor_generator
    maskrcnn_args['mask_head'] = maskrcnn_heads
    maskrcnn_args['mask_predictor'] = mask_predictor
    maskrcnn_args['box_predictor'] = box_predictor
    # Instantiate the segmentation model
    maskrcnn_model = mask_net.maskrcnn_resnet_fpn(backbone_name, truncation, pretrained_backbone=use_pretrained_resnet_backbone, **maskrcnn_args)
    # pretrained?
    print(maskrcnn_model.backbone.out_channels)
    if pretrained_model is not None:
        print("Loading pretrained weights")
        maskrcnn_model.load_state_dict(pretrained_model['model_weights'])
        if pretrained_model['epoch']:
           start_epoch = int(pretrained_model['epoch'])+1
        if 'model_name' in pretrained_model.keys():
           model_name = str(pretrained_model['model_name'])

    # Set to training mode
    print(maskrcnn_model)
    maskrcnn_model.train().to(device)

    optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
    optimizer = torch.optim.Adam(list(maskrcnn_model.parameters()), **optimizer_pars)
    if pretrained_model is not None and 'optimizer_state' in pretrained_model.keys():
       optimizer.load_state_dict(pretrained_model['optimizer_state'])

    start_time = time.time()
    if start_epoch>0:
       num_epochs += start_epoch
    print("Start training, epoch = {:d}".format(start_epoch))
    for e in range(start_epoch, num_epochs):
        train_loss_epoch = main_step("train", e, dataloader_covid_train, optimizer, device, maskrcnn_model, save_every,
                                lrate, model_name, None, None)
        eval_loss_epoch = main_step("eval", e, dataloader_covid_eval, optimizer, device, maskrcnn_model, save_every, lrate, model_name, anchor_generator, save_dir)
        print(
            "Epoch {0:d}: train loss = {1:.3f}, validation loss = {2:.3f}".format(e, train_loss_epoch, eval_loss_epoch))
    end_time = time.time()
    print("Training took {0:.1f} seconds".format(end_time - start_time))


def step(stage, e, dataloader, optimizer, device, model, save_every, lrate, model_name, anchors, save_dir):
    epoch_loss = 0
    for b in dataloader:
        optimizer.zero_grad()
        X, y = b
        if device == torch.device('cuda'):
            X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y[
                'masks'].to(device)
        images = [im for im in X]
        targets = []
        lab = {}
        # THIS IS IMPORTANT!!!!!
        # get rid of the first dimension (batch)
        # IF you have >1 images, make another loop
        # REPEAT: DO NOT USE BATCH DIMENSION
        lab['boxes'] = y['boxes'].squeeze_(0)
        lab['labels'] = y['labels'].squeeze_(0)
        lab['masks'] = y['masks'].squeeze_(0)
        if len(lab['boxes']) > 0 and len(lab['labels']) > 0 and len(lab['masks']) > 0:
            targets.append(lab)
        else:
            pass
        # avoid empty objects
        if len(targets) > 0:
            loss = model(images, targets)
            total_loss = 0
            for k in loss.keys():
                total_loss += loss[k]
            if stage == "train":
                total_loss.backward()
                optimizer.step()
            else:
                pass
            epoch_loss += total_loss.clone().detach().cpu().numpy()
    epoch_loss = epoch_loss / len(dataloader)
    if not (e+1) % save_every and stage == "eval":
        model.eval()
        state = {'epoch': str(e+1), 'model_name':model_name, 'model_weights': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'lrate': lrate, 'anchor_generator':anchors}
        if model_name is None:
            print(save_dir, "mrcnn_covid_segmentation_model_ckpt_" + str(e+1) + ".pth")
            torch.save(state, os.path.join(save_dir, "mrcnn_covid_segmentation_model_ckpt_" + str(e+1) + ".pth"))
        else:
            torch.save(state, os.path.join(save_dir, model_name + "_ckpt_" + str(e+1) + ".pth"))

        model.train()
    return epoch_loss


# run the training of the segmentation algoithm
if __name__ == '__main__':
    config_train = config.get_config_pars("trainval")
    main(config_train, step)



