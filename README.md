# COVID-CT-Mask-Net: Prediction of COVID-19 From CT Scans Using Regional Features

Papers on medrXiv: 

[Lightweight Model For The Prediction of COVID-19 Through The Detection And Segmentation of Lesions in Chest CT Scans](https://www.medrxiv.org/content/10.1101/2020.10.30.20223586v2.full.pdf)

[Detection and Segmentation of Lesion Areas in Chest CT Scans For The Prediction of COVID-19](https://www.medrxiv.org/content/10.1101/2020.10.23.20218461v2.full.pdf)

[COVID-CT-Mask-Net: Prediction of COVID-19 From CT Scans Using Regional Features](https://www.medrxiv.org/content/10.1101/2020.10.11.20211052v2.full.pdf)

Bibtex citation ref: 

```
@article {Ter-Sarkisov2020.10.30.20223586,
	author = {Ter-Sarkisov, Aram},
	title = {Lightweight Model For The Prediction of COVID-19 Through The Detection And Segmentation
	of Lesions in Chest CT Scans},
	elocation-id = {2020.10.30.20223586},
	year = {2020},
	doi = {10.1101/2020.10.30.20223586},
	publisher = {Cold Spring Harbor Laboratory Press},
	journal = {medRxiv}
}

@article {Ter-Sarkisov2020.10.23.20218461,
	author = {Ter-Sarkisov, Aram},
	title = {Detection and Segmentation of Lesion Areas in Chest CT Scans For The Prediction of COVID-19},
	elocation-id = {2020.10.23.20218461},
	year = {2020},
	doi = {10.1101/2020.10.23.20218461},
	publisher = {Cold Spring Harbor Laboratory Press},
	journal = {medRxiv}
}

@article {Ter-Sarkisov2020.10.11.20211052,
	author = {Ter-Sarkisov, Aram},
	title = {COVID-CT-Mask-Net: Prediction of COVID-19 from CT Scans Using Regional Features},
	year = {2020},
	doi = {10.1101/2020.10.11.20211052},
	publisher = {Cold Spring Harbor Laboratory Press},
	journal = {medRxiv}
}
```
## Update 01/11/20
I re-implemented torchvision's segmentation interface locally, in the end it was easier to keep two different files for RPN and RoI for segmentation and classification tasks: `rpn_segmentation, roi_segmentation` vs `roi` and `rpn`. For the validation split in `test_split_segmentation.txt` I get the following results for the two lightweight and two best full models (ResNet50+FPN backbone): 

|  Model	| AP@0.5 	| AP@0.75 	| mAP@[0.5:0.95:0.05] 	| Model size
|:-:	|:-:	|:-:	|:-:|:-:	
| **Lightweight model (truncated ResNet34+FPN)** 	| 59.88% 	| 45.06% 	| 44.76% 	| 11.45M|
| **Lightweight model (truncated ResNet18+FPN)** 	| 49.95% 	| 37.78% 	| 39.32% 	|6.12M|
| **Full model (merged masks)** 	| 61.92% 	| 45.22% 	| 44.68% 	|31.78M|
| **Full model (GGO + C masks)**        |  50.20%| 41.98%|38.71%|31.78M|

The penultimate column is the mean over 10 IoU thresholds, the main metric in the MS COCO leaderboard. 

For each script, two additional arguments were added: `backbone_name`, one of `resnet18, resnet34, resnet50` and `truncation`, one of `0,1,2`. For `resnet50`, only the full (base torchvision model) output is implemented, with 4 connections to FPN. For `resnet18` and `resnet34`, `truncation=0` means use the full backbone model, for `truncation=1` the last block is deleted and `truncation=2` the last two layers are deleted. Only the last layer is connected to the FPN. 

To evaluate the model, run (e.g., for the lightweight with ResNet18+FPN backbone and truncated last block: 
```
python3 evaluation_mean_ap.py --backbone_name resnet18 --ckpt model.pth --mask_type merge --truncation 1 --rpn_nms_th 0.75 --roi_nms_th 0.75 --confidence_th 0.75 
```
To train the segmentation model from scratch to get the results above:
```
python3.5 train_segmentation.py --num_epochs 100 --mask_type merge --save_every 10 --backbone_name resnet18 --truncation 1 --device cuda
```
Results of the classification models derived from the segmentation models above (class sensitivity and overall accuracy):


| Model 	| Control 	| CP 	| COVID-19 	| Overall accuracy
|:-:	|:-:	|:-:	|:-:	|:-:
| **Lightweight model (truncated ResNet34+FPN)** 	|  92.89%	| 91.70% 	| 91.76% 	|92.89%|
| **Lightweight model (truncated ResNet18+FPN)** 	| 96.98% 	| 91.63% 	| 91.35% 	|93.95%|
| **Full model (merged masks)** 	| 97.74% 	| 96.69% 	| 92.68% 	|96.33%|
| **Full model (both masks)** 	| 96.91% 	| 95.06% 	| 93.88% 	|95.64%|

To train a lightweight classifier, you need to specify the backbone name and the truncation level, it must be the same as in the segmentation model from which it is derived. Also, you need to define the size of the RoI batch: roi_batch_size, which is equal to the input in the classification module, number of masks on which the segmentation model was trained `num_class` (2 for merged and 3 for separate) and the number of features in the classification module **S**, `s_features`. You need at least the pretrained weights from a segmentation model. **You cannot train COVID-CT-Mask-Net classifier from scratch.** 
```
python3 train_classifier.py --pretrained_segmentation_model segmentation_model.pth --backbone_name resnet34 --num_epochs 50 --save_every 10 --num_class 2 --truncation 1 --s_features 512 --roi_batch_size 128 --batch_size 8
```
In this case the weights for all parameters except **S** are copied from the segmentation model, all parameters in **S** and weights in the batch normalization layers are updated, but the stats in the batch normalization layers (means and variances) are frozen. After about 50 epochs (2h15min hours on an a GPU with 8Gb VRAM) you should get the model with the accuracy reported above. To evaluate the classifier:
```
python3 evaluate_classifier.py --ckpt classification_model.pth --truncation 1 --num_class 2 --backbone_name resnet34 --roi_batch_size 128 --device cuda --s_features 512
```

## Update 29/10/20
Column 1: Input CT scan slice overlaid with the output of the segmentation model. 

Column 2: Mask maps logit scores (pixel-level) predicted by Mask R-CNN *independently of each other*, i.e. they were output by different RoIs and resized to fit the bounding box prediction. Note COVID-CT-Mask-Net uses a fixed number of RoIs. Only the highest-ranking RoIs are plotted here to avoid the image clutter.

Column 3: ground truth masks for lesions (yellow) and lungs (green, treated as a background).

Column 4: true class (green) and logit scores output by COVID-CT-Mask-Net (red) using the score map's inputs. Note how the classification model learns the distribution and ranking of the regional predictions (bounding boxes and confidence scores) to predict the global (image) class.

<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/segmentation_map_classification_score.png" width="800" height="500" align="center"/>
</p>


## Update 19-22/10/20
I added a large number of updates across all models. Now you can train segmentation and classification models with 3 types of masks: two masks (GGO and C), only GGO and merged GGO and C  masks('lesion'). 

I added methods in the `utils` script to compute the accuracy (mean Average Precision) of Mask R-CNN segmentation models. They are based on matterport's package, but purely in pytorch, no requirements for RLE or pycocotools. A new evaluation script, `evaluation_mean_ap`, which uses these methods for a range of Intersect over Union (IoU) thresholds, has been added too. 

**COVID-CT-Mask-Net (merged masks)**: COVID-19 sensitivity: 93.55%, overall accuracy: 96.33%
|  	| Control 	| CP 	| COVID-19 	|
|:-:	|:-:	|:-:	|:-:	|
| **Control** 	| 9236 	| 188 	| 26 	|
| **CP** 	| 116 	| 7150 	| 129 	|
| **COVID-19** 	| 20 	| 298 	| 4028 	|

**COVID-CT-Mask-Net (GGO + C masks)**: COVID-19 sensitivity: 93.88%, overall accuracy: 95.64%
|  	| Control 	| CP 	| COVID-19 	|
|:-:	|:-:	|:-:	|:-:	|
| **Control** 	| 9158 	| 278 	| 14 	|
| **CP** 	| 204 	| 7030 	| 161 	|
| **COVID-19** 	| 15 	| 251 	| 4080 	|

All segmentation scripts as well as the segmentation dataset interface accept `mask_type` argument, one of `both` (GGO + C), `ggo` (only GGO) and `merge` (merged GGO and C masks). The effect on the size of the model is marginal.       

## 1. Segmentation Model
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/maskrcnncovidsegment.png" width="800" height="250" align="center"/>
</p>
To train and test the model you need Torchvision 0.3.0+

The segmentation model predicts masks of Ground Glass Opacity and Consolidation in CT scans. We trained it on the CNCB CT images with masks (http://ncov-ai.big.ac.cn/download, Experiment data files): 500 training and 150 for testing taken from COVID-positive patients, but some slices have no
lesions. Use the splits in `train_split_segmentation.txt` and `test_split_segmentation.txt` to copy the training data into `covid_data/train/imgs` and `covid_data/train/masks` and test data into `covid_data/test/imgs` and `covid_data/test/masks`. 

Download the pretrained weights into  `pretrained_models/` directory.

To get the inference of the segmentation model, run: 
```
python3.5 inference_segmentation.py --ckpt pretrained_models/segmentation_model_both_classes.pth --test_data_dir covid_data/test --test_imgs_dir imgs --masks_type both
```
This should output predictions like these:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/128_92_with_mask.png" width="600" height="200" align="center"/>
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/133_48_with_mask.png" width="600" height="200" align="center"/>
</p>

For the explanation of plots see the paper. To get the average precision on the data, you also need the mask for each image. For example, for merged masks:
```
python3.5 evaluation_mean_ap.py --ckpt pretrained_weights/segmentation_model_merged_masks.pth --mask_type merge --test_data_dir covid_data/test --test_imgs_dir imgs --gt_dir masks
```
To train the segmentation model, you also need images with masks. Dataset interface `dataset_segmentation.py` converts masks into binary masks with either 2 positive classes (GGO+C) or 1 (GGO only, merged GGO+C). It also extracts labels and bounding boxes that Mask R-CNN requires. 
To train from scratch for the merged masks, run 
```
python3.5 train_segmentation.py --device cuda --num_epochs 100 --use_pretrained_model False -use_pretrained_backbone True --save_every 10 --mask_type merge
```
To get the reported results, and for the COVID-CT-Mask-Net classsifier, we trained the model for 100 epochs (about 4.5 hours on a GPU with 8Gb VRAM).   

## 2. COVID-CT-Mask-Net (Classification Model) 

### 2.1 Full model (ResNet50+FPN backbone)
**The model**
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/covid_ct_mask_net.png" width="800" height="300" align="center"/>
</p>

**Classification module *S***
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/s_module.png" width="300" height="200" align="center"/>
</p>

I reimplemented torchvision's detection library(https://github.com/pytorch/vision/tree/master/torchvision/models/detection) in `/models/mask_net/` with the classification module **s2_new** (**S** in the paper) and other hacks that convert Mask R-CNN into a classification model.
First, download and unpack the CNCB dataset: (http://ncov-ai.big.ac.cn/download), a total of over 100K CT scans. The COVIDx-CT split we used is here: https://github.com/haydengunraj/COVIDNet-CT/blob/master/docs/dataset.md). To extract the COVID-19, pneumonia and normal scans, follow the instructions in the link to COVIDx-CT. You don't need to do any image preprocessing as inthe COVIDNet-CT model. We used the full validation and test split, and a small share of the training data, our sample is in `train_split_classification.txt`. To follow the convention used in the other two datsets, we set Class 0: Control, Class 1: Normal Pneumonia, Class 2: COVID. Thus the dataset interface `datasets/dataset_classification.py` extracts the labels from the file names. The convention for the names must be `[Class]_[PatientID]_[ScanNum]_[SliceNum].png`. To train the classifier, copy the images following this convention into a separate directory, e.g. `train_small`.

### 2.2 Lightweight Models (Truncated ResNet18/34+FPN backbone)

I implemented two backbones, ResNet18 and ResNet34, both with a single FPN module, and two truncations: the last block or two last blocks. 

Backbone model:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/resnet18.png" width="400" height="150" align="center"/>
</p>

The classification model with the lightweight backbone:
<p align="center">
<img src="https://github.com/AlexTS1980/COVID-CT-Mask-Net/blob/master/plots/covid_ct_mask_net_resnet18.png" width="300" height="150" align="center"/>
</p>

Here's the full size comparison:

| Model	| Total #parameters| #Trainable parameters|
|:-:	|:-:	|:-:	|	
| **Lightweight model, 5 blocks, ResNet34+FPN** 	| 24.86M|0.6M|
| **Lightweight model, 4 blocks, ResNet34+FPN** 	| 11.74M|0.6M|
| **Lightweight model, 3 blocks, ResNet34+FPN** 	| 4.92M|0.6M|
| **Lightweight model, 5 blocks, ResNet18+FPN** 	| 14.75M|0.6M|
| **Lightweight model, 4 blocks, ResNet18+FPN** 	| 6.35M|0.6M|
| **Lightweight model, 3 blocks, ResNet18+FPN** 	| 4.25M|0.6M|
|**Full model, 5 blocks, ResNet50+FPN (4 layers)**|34.14M|2.36M|


## 3. Models' hyperparameters

There are two groups of hyperparameters: training (learning rate, weight regularization, optimizer, etc) and Mask R-CNN hyperparameters (Non-max suppression threshold, RPN and RoI batch size, RPN output, RoI score threshold, etc). The ones in the training scripts are the ones we used to get the models in the paper and the results. For the segmentation model you can use any you want, but for COVID-CT-Mask-Net the RoI score threshold (`box_score_thresh`) must be negative (e.g. `-0.01`), because otherwise not all box predictions (`box_detections_per_img`) will be accepted, and the classification module **S** will not get the batch of the right size, hence you will get a tensor mismatch error.

[Update 22/10/20:] Also, our re-implementation of torchvision's Mask R-CNN has a hack that allows maintaining the same batch size regardless of the pre-set `box_score_thresh`. 

For any questions, contact Alex Ter-Sarkisov: (alex.ter-sarkisov@city.ac.uk)
