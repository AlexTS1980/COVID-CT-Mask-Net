Download the pretrained weights, zipped file (~480Mb):

https://drive.google.com/file/d/177dY9jSSCsk-de2pAH9TZnO6TaC56VfX/view?usp=sharing


1. Segmentation weights for two positive classes: `segmentation_model_two_classes.pth`, Ground Glass Opacity and Consolidation segmentation predictions. 

2. Segmentation weights for one class (merged masks): `segmentation_model_merged_masks.pth`, lesion predictions. 

3. Lightweight segmentation model (ResNet34+FPN backbone, truncated last block): `lightweight_segmentation_model_resnet34_t1.pth`, lesion prediction.

4. Lightweight segmentation model (ResNet18+FPN backbone, truncated last block): `lightweight_segmentation_model_resnet18_t1.pth`, lesion prediction.

3. COVID-CT-Mask-Net: `classification_model_two_classes.pth`. The best classification model derived from the segmentation model with two classes. 
I get **95.64%** overall accuracy on the test data, **93.88%** COVID sensitivity on the test split of CNCB CT scans dataset (21192 images).

4. COVID-CT-Mask-Net: `classification_model_merged_masks.pth`.  The best classification model derived from the segmentation model with the merged masks.  
I get **96.33%** overall accuracy on the test data, **92.68%** COVID sensitivity on the test split of CNCB CT scans dataset (21192 images).

5. Lightweight COVID-CT-Mask-Net (ResNet34+FPN backbone): `lightweight_classifier_resnet34_t1.pth` with  11.74M weights. I get **92.89%** overall accuracy on the test data, 
**91.76%** COVID sensitivity on the test split of CNCB CT scans dataset (21192 images).

6. Lightweight COVID-CT-Mask-Net (ResNet18+FPN backbone): `lightweight_classifier_resnet18_t1.pth` with 6.35M weights. I get **93.95%** overall accuracy on the test data, 
**91.35%** COVID sensitivity on the test split of CNCB CT scans dataset (21192 images).  

