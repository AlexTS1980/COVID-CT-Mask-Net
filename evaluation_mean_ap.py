# torchvision:
# /home/enterprise.internal.city.ac.uk/sbrn151/.local/lib/python3.5/site-packages/torchvision/models/detection/__init__.py
import argparse
import os
from collections import OrderedDict

import config_segmentation as config
import torch
import torch.utils.data as data
import torchvision
# implementation of the mAP
import utils
from datasets import dataset_segmentation as dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator


def main(config, step):
    devices = ['cpu', 'cuda']
    mask_classes = ['both', 'ggo', 'merge']
    assert config.device in devices
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    #
    model_name = None
    ckpt = torch.load(config.ckpt, map_location=device)
    if 'model_name' in ckpt.keys():
        model_name = ckpt['model_name']

    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    # get the thresholds
    confidence_threshold, mask_threshold, save_dir, data_dir, ims_dir, gt_dir, mask_type, rpn_nms, roi_nms \
        = config.confidence_th, config.mask_logits_th, config.save_dir, config.test_data_dir, config.test_imgs_dir, \
        config.gt_dir, config.mask_type, config.rpn_nms_th, config.roi_nms_th

    if model_name is None:
        model_name = "maskrcnn_segmentation"
    elif model_name is not None and config.model_name != model_name:
        print("Using model name from the config.")
        model_name = config.model_name

    # either 2+1 or 1+1 classes
    assert mask_type in mask_classes
    if mask_type == "both":
        n_c = 3
    else:
        n_c = 2
    # dataset interface
    dataset_covid_eval_pars = {'stage': 'eval', 'gt': os.path.join(data_dir, gt_dir),
                               'data': os.path.join(data_dir, ims_dir), 'mask_type': mask_type, 'ignore_small':True}
    datapoint_eval_covid = dataset.CovidCTData(**dataset_covid_eval_pars)
    dataloader_covid_eval_pars = {'shuffle': False, 'batch_size': 1}
    dataloader_eval_covid = data.DataLoader(datapoint_eval_covid, **dataloader_covid_eval_pars)
    #
    # MASK R-CNN model
    # Alex: these settings could also be added to the config
    ckpt = torch.load(config.ckpt, map_location=device)
    sizes = ckpt['anchor_generator'].sizes
    aspect_ratios = ckpt['anchor_generator'].aspect_ratios
    anchor_generator = AnchorGenerator(sizes, aspect_ratios)
    print("Anchors: ", anchor_generator)

    # create modules
    box_head = TwoMLPHead(in_channels=7 * 7 * 256, representation_size=128)
    box_predictor = FastRCNNPredictor(in_channels=128, num_classes=n_c)
    mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3], output_size=14, sampling_ratio=2)
    mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=n_c)
    # keyword arguments
    maskrcnn_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 128,
                     'box_nms_thresh': roi_nms, 'box_score_thresh': confidence_threshold, 'rpn_nms_thresh': rpn_nms,
                     'box_head': box_head,
                     'rpn_anchor_generator': anchor_generator, 'mask_roi_pool': mask_roi_pool,
                     'mask_predictor': mask_predictor, 'box_predictor': box_predictor}

    # Instantiate the segmentation model
    maskrcnn_model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False,
                                                                        progress=False, **maskrcnn_args)
    # Load weights
    maskrcnn_model.load_state_dict(ckpt['model_weights'])
    # Set to the evaluation mode
    print(maskrcnn_model)
    maskrcnn_model.eval().to(device)
    # IoU thresholds. By default the model computes AP for each threshold between 0.5 and 0.95 with the step of 0.05
    thresholds = torch.arange(0.5, 1, 0.05).to(device)
    mean_aps_all_th = torch.zeros(thresholds.size()[0]).to(device)
    ap_th = OrderedDict()
    # run the loop for all thresholds
    for t, th in enumerate(thresholds):
        # main method
        ap = step(maskrcnn_model, th, dataloader_eval_covid, device, mask_threshold)
        mean_aps_all_th[t] = ap
        th_name = 'AP@{0:.2f}'.format(th)
        ap_th[th_name] = ap
    print("Done evaluation for {}".format(model_name))
    print("mAP:{0:.2f}".format(mean_aps_all_th.mean().item()))
    for k, aps in ap_th.items():
        print("{0:}:{1:.2f}".format(k, aps))


def compute_map(model, iou_th, dl, device, mask_th):
    mean_aps_this_th = torch.zeros(len(dl), dtype=torch.float)
    for v, b in enumerate(dl):
        x, y = b
        if device == torch.device('cuda'):
            x, y['labels'], y['boxes'], y['masks'] = x.to(device), y['labels'].to(device), y['boxes'].to(device), y[
                'masks'].to(device)
        lab = {'boxes': y['boxes'].squeeze_(0), 'labels': y['labels'].squeeze_(0), 'masks': y['masks'].squeeze_(0)}
        image = [x.squeeze_(0)]  # remove the batch dimension
        out = model(image)
        # scores + bounding boxes + labels + masks
        scores = out[0]['scores']
        bboxes = out[0]['boxes']
        classes = out[0]['labels']
        # remove the empty dimension,
        # output_size x 512 x 512
        predict_mask = out[0]['masks'].squeeze_(1) > mask_th
        if len(scores) > 0 and len(lab['labels']) > 0:
            # threshold for the masks:
            ap, _, _, _ = utils.compute_ap(lab['boxes'], lab['labels'], lab['masks'], bboxes, classes, scores,
                                           predict_mask, iou_threshold=iou_th)
            mean_aps_this_th[v] = ap
        elif not len(scores) and not len(lab['labels']):
            mean_aps_this_th[v] = 1
        elif not len(scores) and len(lab['labels']) > 0:
            continue
        elif len(scores) > 0 and not len(y['labels']):
            continue
    return mean_aps_this_th.mean().item()


if __name__ == "__main__":
    config_mean_ap = config.get_config_pars("precision")
    main(config_mean_ap, compute_map)
