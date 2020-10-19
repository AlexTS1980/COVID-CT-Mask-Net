import os

import torch
import torch.utils.data as data
import torchvision
import utils
from datasets import dataset_segmentation as dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.mask_rcnn import maskrcnn_resnet50_fpn
from torchvision.models.detection.rpn import AnchorGenerator

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
ct_classes = {0: '__bgr', 1: 'GGO', 2: 'C'}

data_dir = 'imgs'
gt_dir = 'masks'
dataset_covid_eval_pars = {'stage': 'eval', 'gt': os.path.join('../covid_data/test', gt_dir),
                           'data': os.path.join('../covid_data/test', data_dir)}
datapoint_eval_covid = dataset.CovidCTData(**dataset_covid_eval_pars)

dataloader_covid_eval_pars = {'shuffle': False, 'batch_size': 1}
dataloader_eval_covid = data.DataLoader(datapoint_eval_covid, **dataloader_covid_eval_pars)
weights = 'segmentation_model.pth'
covid_detector_weights = torch.load(os.path.join("pretrained_weights", weights), map_location="cpu")
# keyword arguments
inference_args = {'num_classes': None, 'min_size': 512, 'max_size': 1024, 'box_detections_per_img': 128,
                  'box_nms_thresh': 0.25, 'box_score_thresh': .75, 'rpn_nms_thresh': 0.25}
print(inference_args)
# many small anchors
anchor_generator = AnchorGenerator(
    sizes=tuple([(2, 4, 8, 16, 32) for r in range(5)]),
    aspect_ratios=tuple([(0.1, 0.25, 0.5, 1, 1.5, 2) for rh in range(5)]))

box_head = TwoMLPHead(in_channels=7 * 7 * 256, representation_size=128)
box_predictor = FastRCNNPredictor(in_channels=128, num_classes=3)
mask_roi_pool = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0, 1, 2, 3], output_size=14, sampling_ratio=2)
mask_predictor = MaskRCNNPredictor(in_channels=256, dim_reduced=256, num_classes=3)

inference_args['box_head'] = box_head
inference_args['rpn_anchor_generator'] = anchor_generator
inference_args['mask_roi_pool'] = mask_roi_pool
inference_args['mask_predictor'] = mask_predictor
inference_args['box_predictor'] = box_predictor

maskrcnn_model = maskrcnn_resnet50_fpn(pretrained=False, **inference_args)
maskrcnn_model.load_state_dict(covid_detector_weights['model_weights'])
maskrcnn_model.eval()
maskrcnn_model = maskrcnn_model.to(device)
thresholds = torch.arange(0.5, 1, 0.05).to(device)
mean_aps_all_th = torch.zeros(thresholds.size()[0]).to(device)


def compute_map(iou_th):
    mean_aps_this_th = torch.zeros(len(dataloader_eval_covid), dtype=torch.float)
    for id, b in enumerate(dataloader_eval_covid):
        X, y = b
        if device == torch.device('cuda'):
            X, y['labels'], y['boxes'], y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y[
                'masks'].to(device)
        images = [im for im in X]
        lab = {}
        # THIS IS IMPORTANT!!!!!
        # get rid of the first dimension (batch)
        # IF you have >1 images, make another loop
        # REPEAT: DO NOT USE BATCH DIMENSION
        lab['boxes'] = y['boxes'].squeeze_(0)
        lab['labels'] = y['labels'].squeeze_(0)
        lab['masks'] = y['masks'].squeeze_(0)
        image = [X.squeeze_(0)]  # remove the batch dimension
        out = maskrcnn_model(image)
        # scores + bounding boxes + labels + masks
        scores = out[0]['scores']
        bboxes = out[0]['boxes']
        classes = out[0]['labels']
        # remove the empty dimension,
        # output_size x 512 x 512
        predict_mask = out[0]['masks'].squeeze_(1) > 0.5
        if len(scores) > 0 and len(lab['labels']) > 0:
            # threshold for the masks:
            mAP, _, _, _ = utils.compute_ap(lab['boxes'], lab['labels'], lab['masks'], bboxes, classes, scores,
                                            predict_mask, iou_threshold=iou_th)
            mean_aps_this_th[id] = mAP
        elif not len(scores) and not len(lab['labels']):
            mean_aps_this_th[id] = 1
        elif not len(scores) and len(lab['labels']) > 0:
            continue
        elif len(scores) > 0 and not len(y['labels']):
            continue
    print("AP @{0:.2f}:{1:.2f}".format(iou_th.item(), mean_aps_this_th.mean().item()))
    return mean_aps_this_th, mean_aps_this_th.mean().item()


for t, th in enumerate(thresholds):
    aps, map = compute_map(th)
    mean_aps_all_th[t] = map

print("mAP:{0:.2f}".format(mean_aps_all_th.mean().item()))
