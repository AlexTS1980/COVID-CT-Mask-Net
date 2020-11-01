# COVID-CT-Mask-Net
# Torchvision detection package is locally re-implemented
# Transformed into a classification model with Mask R-CNN backend
# by Alex Ter-Sarkisov@City, University of London
# alex.ter-sarkisov@city.ac.uk
# 2020
import os
import re
import sys
import time
import config_classifier as config
import cv2
#######################################
import models.mask_net as mask_net
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import utils
from PIL import Image as PILImage
from models.mask_net.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from models.mask_net.rpn import AnchorGenerator



def main(config, step):
    torch.manual_seed(time.time())
    start_time = time.time()
    devices = ['cpu', 'cuda']
    backbones = ['resnet50', 'resnet34', 'resnet18']
    truncation_levels = ['0','1','2']

    assert config.device in devices
    assert config.backbone_name in backbones
    assert config.truncation in truncation_levels

    pretrained_model, model_name, test_data_dir, device, rpn_nms, roi_nms, backbone_name, truncation, roi_batch_size, n_c, s_features\
              = config.ckpt, config.model_name, config.test_data_dir, config.device, config.rpn_nms_th, \
                config.roi_nms_th, config.backbone_name, config.truncation, config.roi_batch_size, config.num_classes, config.s_features

    if torch.cuda.is_available() and device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # either 2+1 or 1+1 classes
    ckpt = torch.load(pretrained_model, map_location=device)
    # 'box_detections_per_img': batch size input in module S
    # 'box_score_thresh': negative to accept all predictions
    covid_mask_net_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': roi_batch_size,
                           'box_nms_thresh': roi_nms, 'box_score_thresh': -0.01, 'rpn_nms_thresh': rpn_nms}

    print(covid_mask_net_args)
    # extract anchor generator from the checkpoint
    sizes = ckpt['anchor_generator'].sizes
    aspect_ratios = ckpt['anchor_generator'].aspect_ratios
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    # Faster R-CNN interfaces, masks not implemented at this stage
    box_head = TwoMLPHead(in_channels=256*7*7, representation_size=128)
    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
    # Mask prediction is not necessary, keep it for future extensions
    covid_mask_net_args['rpn_anchor_generator'] = anchor_generator
    covid_mask_net_args['box_predictor'] = box_predictor
    covid_mask_net_args['box_head'] = box_head
    # representation size of the S classification module
    # these should be provided in the config
    covid_mask_net_args['s_representation_size'] = s_features
    # Instance of the model, copy weights
    covid_mask_net_model = mask_net.fasterrcnn_resnet_fpn(backbone_name, truncation, **covid_mask_net_args)
    covid_mask_net_model.load_state_dict(ckpt['model_weights'])
    covid_mask_net_model.eval().to(device)
    print(covid_mask_net_model)
    # confusion matrix
    confusion_matrix = torch.zeros(3, 3, dtype=torch.int32).to(device)

    for idx, f in enumerate(os.listdir(test_data_dir)):
        step(f, covid_mask_net_model, test_data_dir, device, confusion_matrix)

    print("------------------------------------------")
    print("Confusion Matrix for 3-class problem:")
    print("0: Control, 1: Normal Pneumonia, 2: COVID")
    print(confusion_matrix)
    print("------------------------------------------")
    # confusion matrix
    cm = confusion_matrix.float()
    cm[0, :].div_(cm[0, :].sum())
    cm[1, :].div_(cm[1, :].sum())
    cm[2, :].div_(cm[2, :].sum())
    print("------------------------------------------")
    print("Class Sensitivity:")
    print(cm)
    print("------------------------------------------")
    print('Overall accuracy:')
    print(confusion_matrix.diag().float().sum().div(confusion_matrix.sum()))
    end_time = time.time()
    print("Evaluation took {0:.1f} seconds".format(end_time - start_time))

def test_step(im_input, model, source_dir, device, c_matrix):
    # CNCB NCOV datasets: the first integer is the correct class:
    # 0: control
    # 1: pneumonia
    # 2: COVID
    # extract the correct class from the file name
    correct_class = int(im_input.split('/')[-1].split('_')[0])
    im = PILImage.open(os.path.join(source_dir, im_input))
    if im.mode != 'RGB':
        im = im.convert(mode='RGB')
    # get rid of alpha channel
    img = np.array(im)
    # print(img)
    if img.shape[2] > 3:
        img = img[:, :, :3]

    t_ = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(512),
        transforms.ToTensor()])
    img = t_(img)
    if device == torch.device('cuda'):
        img = img.to(device)
    out = model([img])
    pred_class = out[0]['final_scores'].argmax().item()
    # get confusion matrix
    c_matrix[correct_class, pred_class] += 1


# run the inference
if __name__ == '__main__':
    config_test = config.get_config_pars_classifier("test")
    main(config_test, test_step)
