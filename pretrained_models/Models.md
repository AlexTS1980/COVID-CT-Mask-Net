Download the two pretrained weights, zipped file (~300Mb):

https://drive.google.com/drive/folders/17j8LjLmCyIe7HfzN_uVrapiRgqrXJWWY?fbclid=IwAR1QSr5ucgYAbT1HQnn4TW3fA1eudW5pYy7PfwTMHj2T5WMzn33ehh8FOj0

Both models are introduced in the paper.

1. Segmentation weights: segmentation_model.pth, Ground Glass Opacity and Consolidation segmentation predictions. 

2. COVID-CT-Mask-Net: classification_model.pth. This is the weights for the model in which the classification head+batch normalization weights were trained (3.5M in total), while batch normalization history (means+variances) were frozen. 
I get 90.66% overall accuracy on the test data, 90.80% COVID sensitivity on the test split of CNCB CT scans dataset (20182 images).



