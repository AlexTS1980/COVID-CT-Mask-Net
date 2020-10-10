# SEGMENTATION MODEL CONFIG
import argparse
import utils


def get_config_pars(stage):
    parser_ = argparse.ArgumentParser(
        description='arguments for training Mask R-CNN segmentation for 2 classes '
                    '(Ground Glass Opacity and Consolidation) on CNCB dataset')

    parser_.add_argument("--device", type=str, default='cpu')
    parser_.add_argument("--model_name", type=str, default=None)
    
    if stage == "trainval":
        parser_.add_argument("--start_epoch", type=str, default=0)
        parser_.add_argument("--model", type=str, help="Pretrained model, nust be a checkpoint with keys: model_weights, optimizer_state, anchor_generator", default=None)
        parser_.add_argument("--num_epochs", type=int, default=50)
        parser_.add_argument("--use_pretrained_model", type=utils.str_to_bool, default=False)
        parser_.add_argument("--use_pretrained_resnet_backbone", type=utils.str_to_bool, default=False,
                             help="Use the ResNet50w/FPN weights from Torchvision repository")
        parser_.add_argument("--save_dir", type=str, default="saved_models",
                             help="Directory to save checkpoints")
        parser_.add_argument("--train_data_dir", type=str, default='covid_data/train',
                             help="Path to the training data. Must contain images and binary masks")
        parser_.add_argument("--val_data_dir", type=str, default='covid_data/test',
                             help="Path to the validation data. Must contain images and binary masks")
        parser_.add_argument("--gt_dir", type=str, default='masks',
                             help="Path to directory with binary masks. Must be in the data directory.")
        parser_.add_argument("--imgs_dir", type=str, default='imgs')
        parser_.add_argument("--batch_size", type=int, default=1, help="Implemented only for batch size = 1")
        parser_.add_argument("--save_every", type=int, default=10)
        parser_.add_argument("--lrate", type=float, default=1e-5, help="Learning rate")

    elif stage == "test":
        parser_.add_argument("--ckpt", type=str,
                             help="Checkpoint file in .pth format. "
                                  "Must contain the following keys: model_weights, optimizer_state, anchor_generator")
        parser_.add_argument("--test_data_dir", type=str, default='covid_data/test',
                             help="Path to the test data. Must contain images and may contain binary masks")
        parser_.add_argument("--test_imgs_dir", type=str, default='imgs', help="Directory with images. "
                                                                               "Must be in the test data directory.")
        parser_.add_argument("--save_dir", type=str, default='eval_imgs', help="Directory to save "
                                                                               "segmentation results.")
        parser_.add_argument("--confidence_th", type=float, default=0.75, help="Lower confidence score threshold on positive "
                                                                   "prediction from RoI.")
        parser_.add_argument("--mask_logits_th", type=float, default=0.5, help="Lower threshold for positve mask prediction at "
                                                                   "pixel level.")

    model_args = parser_.parse_args()
    return model_args
