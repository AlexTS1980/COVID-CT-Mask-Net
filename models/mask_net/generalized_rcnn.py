# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """
    # Alex: 
    # s2: classification module S
    def __init__(self, backbone, rpn, roi_heads, transform, s2):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.s2new = s2

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        # Alex
        # Segmentation step is the same as in torchvision
        if self.model_type=="segmentation":
           proposals, proposal_losses = self.rpn(images, features, targets)
           detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
           detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

           losses = {}
           losses.update(detector_losses)
           losses.update(proposal_losses)

           if self.training:
              return losses

           return detections
        # Classification: compute only the loss in the S module
        else:
           proposals = self.rpn(images, features, targets)
           detections = self.roi_heads(features, proposals, images.image_sizes, targets)
           scores_covid_boxes = self.s2new(detections[0]['ranked_boxes'])
           scores_covid_img = [dict(final_scores=scores_covid_boxes.squeeze_(0))]
           return scores_covid_img
