# SEGMENTATION MODEL CONFIG
import argparse

import utils


def get_config_pars_classifier(stage):
    parser_ = argparse.ArgumentParser(
        description='arguments for training COVID-CT-Mask-Net on CNCB dataset')

    parser_.add_argument("--device", type=str, default='cpu')
    parser_.add_argument("--model_name", type=str, default=None)
    parser_.add_argument("--num_classes", type=int, default=None, help="This refers to the number of classes in the segmentation mode, so either 2 or 3")
    parser_.add_argument("--rpn_nms_th", type=float, default=0.75, help="Both at train and test stages")
    parser_.add_argument("--roi_nms_th", type=float, default=0.75, help="Both at train and test stages")
    parser_.add_argument("--backbone_name", type=str, default='resnet50', help="One of resnet50, resnet34, reesnet18")
    parser_.add_argument("--truncation", type=str, default="0", help="One of 0,1,2 for no truncation, last block, two last blocks")
    parser_.add_argument("--roi_batch_size", type=int, default=256, help="RoI batch size output, input in the S classifier module")
    parser_.add_argument("--s_features", type=int, default=None, help="Number of features in the S classification module")

    if stage == "trainval":
        parser_.add_argument("--start_epoch", type=str, default=0)
        parser_.add_argument("--num_epochs", type=int, default=50)
        parser_.add_argument("--pretrained_classification_model", type=str, default=None)
        parser_.add_argument("--pretrained_segmentation_model", type=str, default=None,
                             help="Either this or pretrained classifier must be defined!")
        parser_.add_argument("--save_dir", type=str, default="saved_models",
                             help="Directory to save checkpoints")
        parser_.add_argument("--train_data_dir", type=str, default='../covid_data/cncb/train_large',
                             help="Path to the training data. Must contain images and binary masks")
        parser_.add_argument("--val_data_dir", type=str, default='../covid_data/cncb/new_/val',
                             help="Path to the validation data. Must contain images and binary masks")
        parser_.add_argument("--batch_size", type=int, default=8, help="Implemented only for batch size = 1")
        parser_.add_argument("--save_every", type=int, default=10)
        parser_.add_argument("--lrate", type=float, default=1e-5, help="Learning rate")

    elif stage == "test":
        parser_.add_argument("--ckpt", type=str,
                             help="Checkpoint file in .pth format. "
                                  "Must contain the following keys: model_weights, optimizer_state, anchor_generator")
        parser_.add_argument("--test_data_dir", type=str, default='../covid_data/cncb/ncov-ai.big.ac.cn/download/test',
                             help="Path to the test data. Must contain images and may contain binary masks")

    model_args = parser_.parse_args()
    return model_args
