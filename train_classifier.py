# COVID-CT-Mask-Net
# I re-implemented Torchvision's detection library (Faster and Mask R-CNN) as a classifier
# Alex Ter-Sarkisov @ City, University of London
# alex.ter-sarkisov@city.ac.uk
#
import os
import pickle
import sys
import sys
import time

import config_classifier
import cv2
import datasets.dataset_classifier as dataset
# IMPORT LOCAL IMPLEMENTATION OF TORCHVISION'S DETECTION LIBRARY
import models.mask_net as mask_net
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import utils
from PIL import Image as PILImage
# I renamed Mask R-CNN interface to COVID_MASK_NET for convenience
from models.mask_net.covid_mask_net import MaskRCNNPredictor
from models.mask_net.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from models.mask_net.rpn import AnchorGenerator
from torch.utils import data
from torchvision import transforms


# main method
def main(config, main_step):
    torch.manual_seed(time.time())
    start_time = time.time()
    devices = ['cpu', 'cuda']
    updates = ['heads', 'heads_bn', 'full']
    start_epoch, update_type, pretrained_classifier, pretrained_segmenter, model_name, num_epochs, save_dir, train_data_dir, val_data_dir, \
    batch_size, device, save_every, lrate = config.start_epoch, config.update_type, config.pretrained_classification_model, \
                                            config.pretrained_segmentation_model, \
                                            config.model_name, config.num_epochs, config.save_dir, \
                                            config.train_data_dir, config.val_data_dir, \
                                            config.batch_size, config.device, config.save_every, config.lrate

    if pretrained_classifier is not None and pretrained_segmenter is not None:
        print("Not clear which model to use, switching to the classifier")
        pretrained_model = pretrained_classifier
    elif pretrained_classifier is not None and pretrained_segmenter is None:
        pretrained_model = pretrained_classifier
    else:
        pretrained_model = pretrained_segmenter

    assert device in devices
    assert update_type in updates
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    ##############################################################################################
    # DATASETS+DATALOADERS
    # Alex: could be added in the config file in the future
    # parameters for the dataset
    # 512x512 is the recommended image size input
    dataset_covid_pars_train_cl = {'stage': 'train', 'data': train_data_dir, 'img_size': 512}
    datapoint_covid_train_cl = dataset.COVID_CT_DATA(**dataset_covid_pars_train_cl)
    #
    dataset_covid_pars_eval_cl = {'stage': 'eval', 'data': val_data_dir, 'img_size': 512}
    datapoint_covid_eval_cl = dataset.COVID_CT_DATA(**dataset_covid_pars_eval_cl)
    #
    dataloader_covid_pars_train_cl = {'shuffle': True, 'batch_size': batch_size}
    dataloader_covid_train_cl = data.DataLoader(datapoint_covid_train_cl, **dataloader_covid_pars_train_cl)
    #
    dataloader_covid_pars_eval_cl = {'shuffle': True, 'batch_size': batch_size}
    dataloader_covid_eval_cl = data.DataLoader(datapoint_covid_eval_cl, **dataloader_covid_pars_eval_cl)
    #
    ##### LOAD PRETRAINED WEIGHTS FROM MASK R-CNN MODEL
    # This must be the full path to the checkpoint with the anchor generator and model weights
    # Assumed that the keys in the checkpoint are model_weights and anchor_generator
    ckpt = torch.load(pretrained_model, map_location=device)
    # keyword arguments
    # box_score_threshold:negative!
    # set both NMS thresholds to 0.75 to get adjacent RoIs
    # Box detections/image: batch size for the classifier
    #
    covid_mask_net_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 256,
                           'box_nms_thresh': 0.75, 'box_score_thresh': -0.01, 'rpn_nms_thresh': 0.75}

    # copy the anchor generator parameters, create a new one to avoid implementations' clash
    sizes = ckpt['anchor_generator'].sizes
    aspect_ratios = ckpt['anchor_generator'].aspect_ratios
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    # out_channels:256
    # num_classes:3 (1+2)
    box_head_input_size = 256 * 7 * 7
    box_head = TwoMLPHead(in_channels=box_head_input_size, representation_size=128)
    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=3)
    # Mask prediction is not necessary, keep it for future extensions
    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=3)

    covid_mask_net_args['rpn_anchor_generator'] = anchor_generator
    covid_mask_net_args['mask_predictor'] = mask_predictor
    covid_mask_net_args['box_predictor'] = box_predictor
    covid_mask_net_args['box_head'] = box_head

    covid_mask_net_model = mask_net.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, progress=False,
                                                          **covid_mask_net_args)

    # which parameters to train?
    trained_pars = []

    if pretrained_classifier is None:
        for _n, _par in covid_mask_net_model.state_dict().items():
            if _n in ckpt['model_weights']:
                print('Loading parameter', _n)
                _par.copy_(ckpt['model_weights'][_n])
    else:
        covid_mask_net_model.load_state_dict(ckpt['model_weights'])
        if ckpt['epoch']:
            start_epoch = int(ckpt['epoch'])
        if ckpt['model_name']:
            model_name = ckpt['model_name']

    # Evaluation mode, no labels!
    covid_mask_net_model.eval()
    # set the model to training mode without triggering the 'training' mode of Mask R-CNN
    utils.switch_model_on(covid_mask_net_model, trained_pars, update_type)
    utils.set_to_train_mode(covid_mask_net_model, update_type)
    print(covid_mask_net_model)
    covid_mask_net_model = covid_mask_net_model.to(device)

    total_trained_pars = sum([x.numel() for x in trained_pars])
    print("Total trained pars {0:d}".format(total_trained_pars))

    optimizer_pars = {'lr': 1e-5, 'weight_decay': 1e-3}
    optimizer = torch.optim.Adam(trained_pars, **optimizer_pars)
    if pretrained_classifier is not None and 'optimizer_state' in ckpt.keys():
        optimizer.load_state_dict(ckpt['optimizer_state'])

    for e in range(start_epoch, num_epochs):
        train_loss_epoch = main_step("train", e, dataloader_covid_train_cl, optimizer, device, covid_mask_net_model,
                                     save_every, lrate, model_name, None, None, update_type=update_type)
        eval_loss_epoch = main_step("eval", e, dataloader_covid_eval_cl, optimizer, device, covid_mask_net_model,
                                    save_every, lrate, model_name, anchor_generator, save_dir, update_type=update_type)
        print(
            "Epoch {0:d}: train loss = {1:.3f}, validation loss = {2:.3f}".format(e, train_loss_epoch, eval_loss_epoch))
    end_time = time.time()
    print("Training took {0:.1f} seconds".format(end_time - start_time))


def step(stage, e, dataloader, optimizer, device, model, save_every, lrate, model_name, anchors, save_dir, update_type):
    epoch_loss = 0
    for id, b in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = b
        print(id, X.size(), e, stage)
        if device == torch.device('cuda'):
            X, y = X.to(device), y.to(device)
        # some batches are less than batch_size
        batch_s = X.size()[0]
        batch_scores = []
        # input all images in the batch into COVID-Mask-Net to get B scores
        for id in range(batch_s):
            image = [X[id]]  # remove the batch dimension
            predict_scores = model(image)
            batch_scores.append(predict_scores[0]['final_scores'])
        # batchify scores/image and compute binary cross-entropy loss
        batch_scores = torch.stack(batch_scores)
        batch_loss = F.binary_cross_entropy_with_logits(batch_scores, y)
        print(batch_loss)
        if stage == "train":
            batch_loss.backward()
            optimizer.step()
        else:
            pass
        epoch_loss += batch_loss.clone().detach().cpu().numpy()
    epoch_loss = epoch_loss / len(dataloader)
    if not e % save_every and stage == "eval":
        model.eval()
        state = {'epoch': str(e), 'model_weights': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'lrate': lrate, 'anchor_generator': anchors,
                 'model_name': model_name}
        if model_name is None:
            torch.save(state, os.path.join(save_dir, "covid_ct_mask_net" + str(e) + ".pth"))
        else:
            torch.save(state, os.path.join(save_dir, model_name + "_ckpt_" + str(e) + ".pth"))
        utils.set_to_train_mode(model, update_type)
    return epoch_loss


# run the training
if __name__ == '__main__':
    config_train = config_classifier.get_config_pars_classifier("trainval")
    if config_train.pretrained_classification_model is None and config_train.pretrained_segmentation_model is None:
        print("You must have at least one pretrained model!")
        sys.exit(0)
    else:
        main(config_train, step)
