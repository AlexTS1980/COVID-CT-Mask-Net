# COVID-CT-Mask-Net: Prediction of COVID-19 From CT Scans Using Regional Features

## 1. Segmentation Model
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/maskrcnncovidsegment.png" width="800" height="250" align="center"/>
</p>
To train and test the model you need Torchvision 0.3.0+

The segmentation model predicts masks of Ground Glass Opacity and Consolidation in CT scans. We trained it on the CNCB CT images with masks (http://ncov-ai.big.ac.cn/download, Experiment data files): 500 training and 150 for testing taken from COVID-positive patients, but some slices have no
lesions. Use the splits in `train_split_segmentation.txt` and `test_split_segmentation.txt` to copy the training data into `covid_data/train/imgs` and `covid_data/train/masks` and test data into `covid_data/test/imgs` and `covid_data/test/masks`. 

Download the pretrained weights into  `pretrained_models/` directory.

To get the inference, run: 
```
python3.5 inference_segmentation.py --ckpt pretrained_models/segmentation_model.pth --test_data_dir covid_data/test --test_imgs_dir imgs
```
This should output predictions like these:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/128_92_with_mask.png" width="600" height="200" align="center"/>
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/133_48_with_mask.png" width="600" height="200" align="center"/>
</p>

For the explanation of plots see the paper,. To train the model, you also need images with masks. Dataset interface `/datasets/dataset_segmentation.py` converts masks into binary masks for 2 classes: Ground Glass Opacity and Consolidation. It also extracts labels and bounding boxes that Mask R-CNN requires. 
To train from scratch, run 

```
python3.5 train_segmentation.py --device cuda --num_epochs 50 --use_pretrained_model False -use_pretrained_backbone True --save_every 10
```
For the COVID-CT-Mask-Net classsifier, we trained the model for 50 epochs (about 3 hours on a GPU with 8Gb VRAM). For other arguments see `config_segmentation.py`.  

## 2. COVID-CT-Mask-Net (Classification Model) 

**The model**
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/covid_ct_mask_net.png" width="800" height="300" align="center"/>
</p>

**Classification module *S***
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/s_module.png" width="300" height="200" align="center"/>
</p>

I reimplemented torchvision's detection library(https://github.com/pytorch/vision/tree/master/torchvision/models/detection) in `/models/mask_net/` with the classification module **s2_new** (**S** in the paper) and other hacks that convert Mask R-CNN into a classification model.
First, download and unpack the CNCB dataset: (http://ncov-ai.big.ac.cn/download), a total of over 100K CT scans. The COVIDx-CT split we used is here: https://github.com/haydengunraj/COVIDNet-CT/blob/master/docs/dataset.md). To extract the COVID, pneumonia and normal scans, follow the instructions in the link to COVIDx-CT. You don't need to do any image preprocessing as inthe COVIDNet-CT model. We used the full validation and test split, and a small share of the training data, our sample is in `train_split_classification.txt`. To follow the convention used in the other two datsets, we set Class 0: Control, Class 1: Normal Pneumonia, Class 2: COVID. Thus the dataset interface `datasets/dataset_classification.py` extracts the labels from the file names. To evaluate the pretrained model, run

```
python3.5 evaluate_classifier.py --ckpt pretrained_models/segmentation.pth --test_data_dir covid_data/cncb/test
```
You should get about **90.80%** COVID sensitivity and **91.66%** overall accuracy. 

To train the model, copy the images in `train_split_classification.txt` into a separate folder (e.g. `train_small`). You need at least the pretrained weights from a segmentation model, such as `segmentation_model.pth`. You cannot train it from scratch.
```
python3 train_classifier.py --pretrained_segmentation_model pretrained_models/segmentation_model.pth --train_data_dir train_small --num_epochs 50 --save_every 10 --update_type heads_bn --batch_size 8 --device cuda
```
In this case the wieghts for all parameters except **S** are copied from the segmentation model, all parameters in **S** and weights in the batch normalization layers are updated, but the stats in the batch normalization layers (means and variances) are frozen. For other arguments see `config_classifier.py`. After about 40 epochs (6 hours on an a GPU with 8Gb VRAM) you should get the model with the accuracy like the one in 'classification_model.pth'. The full confusion matrix of the reported model (rows: True, columns: predicted):

|  	| Control 	| CP 	| COVID 	|
|:-:	|:-:	|:-:	|:-:	|
| **Control** 	| 8703 	| 738 	| 9 	|
| **CP** 	| 406 	| 6767 	| 213 	|
| **COVID** 	| 14 	| 386 	| 3946 	|

## 3. Models' hyperparameters

There are two groups of hyperparameters: training (learning rate, weight regularization, optimizer, etc) and Mask R-CNN hyperparameters (Non-max suppression threshold, RPN and RoI batch size, RPN output, RoI score threshold, etc). The ones in the training scripts 
are the ones we used to get the models in the paper and the results. For the segmentation model you can use any you want, but for COVID-CT-Mask-Net the RoI score threshold (`box_score_thresh`) must be negative (e.g. `-0.01`), because otherwise not all box predictions 
will be accepted, and the classification module **S** will not get the batch of the right size, hence you will get a tensor mismatch error.

