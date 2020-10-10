# COVID Mask R-CNN project
# You must have Torchvision v0.3.0+
#
import argparse
import time
import pickle
import torch
import torchvision
import numpy as np
import os, sys
import cv2
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign
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
    
    # use pretrained?
    use_pretrained_model = config.use_pretrained_model
    pretrained_model = config.model
    if use_pretrained_model and pretrained_model is None:
        print("Model not provided, training from scratch")
        use_pretrained_model = False
    if not use_pretrained_model and model is not None:
        print("It seems you want to load the weights")
        use_pretrained_model = True
        backbone=False 
    # 
    if use_pretrained_model:
       model = torch.load(pretrained_model)
    # import arguments from the config file
    start_epoch, model_name, backbone, num_epochs, save_dir, train_data_dir, val_data_dir, imgs_dir, gt_dir, batch_size, device, save_every, lrate = \
        config.start_epoch, config.model_name, config.use_pretrained_resnet_backbone, config.num_epochs, config.save_dir, \
        config.train_data_dir, config.val_data_dir, config.imgs_dir, config.gt_dir, config.batch_size, config.device, config.save_every, config.lrate

    if use_pretrained_model:
       backbone=False
     
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
    torch.manual_seed(time.time())
    ##############################################################################################
    # DATASETS+DATALOADERS
    # Alex: could be added in the config file in the future
    # parameters for the dataset
    dataset_covid_pars_train = {'stage': 'train', 'gt': os.path.join(train_data_dir, gt_dir),
                                'data': os.path.join(train_data_dir, imgs_dir)}
    datapoint_covid_train = dataset.CovidCTData(**dataset_covid_pars_train)

    dataset_covid_pars_eval = {'stage': 'eval', 'gt': os.path.join(val_data_dir, gt_dir),
                               'data': os.path.join(val_data_dir, imgs_dir)}
    datapoint_covid_eval = dataset.CovidCTData(**dataset_covid_pars_eval)
    ###############################################################################################
    dataloader_covid_pars_train = {'shuffle': True, 'batch_size': batch_size}
    dataloader_covid_train = data.DataLoader(datapoint_covid_train, **dataloader_covid_pars_train)
    #
    dataloader_covid_pars_eval = {'shuffle': True, 'batch_size': batch_size}
    dataloader_covid_eval = data.DataLoader(datapoint_covid_eval, **dataloader_covid_pars_eval)
    ###############################################################################################
    # MASK R-CNN model
    # Alex: these settings could also be added to the config
    maskrcnn_args = {'min_size': 512, 'max_size': 1024, 'rpn_batch_size_per_image': 1024, 'rpn_positive_fraction': 0.75,
                     'box_positive_fraction': 0.75, 'box_fg_iou_thresh': 0.75, 'box_bg_iou_thresh': 0.5,
                     'num_classes': None, 'box_batch_size_per_image': 1024, 'box_nms_thresh': 0.75,
                     'rpn_nms_thresh': 0.75}

    # Alex: for Ground glass opacity and consolidatin segmentation
    # many small anchors
    # use all outputs of FPN
    # IMPORTANT!! For the pretrained weights, this determines the size of the anchor layer in RPN!!!!
    # pretrained model must have anchors
    if not use_pretrained_model:
       anchor_generator = AnchorGenerator(
           sizes=tuple([(2, 4, 8, 16, 32) for r in range(5)]),
           aspect_ratios=tuple([(0.1, 0.25, 0.5, 1, 1.5, 2) for rh in range(5)]))
    else:
       sizes = model['anchor_generator'].sizes
       aspect_ratios = model['anchor_generator'].aspect_ratios
       anchor_generator = AnchorGenerator(sizes, aspect_ratios)

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
    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=backbone,
                                                                        progress=True, **maskrcnn_args)
    # pretrained?
    if use_pretrained_model:
        maskrcnn_model.load_state_dict(model['model_weights'])
        if model['epoch']:
           start_epoch = int(model['epoch'])
        if model['model_name']:
           model_name = model['model_name']

    # Set to training mode
    print(maskrcnn_model)
    maskrcnn_model.train().to(device)

    optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
    optimizer = torch.optim.Adam(list(maskrcnn_model.parameters()), **optimizer_pars)
    if use_pretrained_model and model['optimizer_state']:
       optimizer.load_state_dict(model['optimizer_state'])

    start_time = time.time()

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
    if not e % save_every and stage == "eval":
        model.eval()
        state = {'epoch': str(e), 'model_name':model_name, 'model_weights': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'lrate': lrate, 'anchor_generator':anchors}
        if model_name is None:
            torch.save(state, os.path.join(save_dir, "mrcnn_covid_segmentation_model_ckpt_" + str(e) + ".pth"))
        else:
            torch.save(state, os.path.join(save_dir, model_name + "_ckpt_" + str(e) + ".pth"))

        model.train()
    return epoch_loss


# run the training of the segmentation algoithm
if __name__ == '__main__':
    config_train = config.get_config_pars("trainval")
    main(config_train, step)



