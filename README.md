# AnalyticsVidhya_GameOfDeepLearning

This repository contains **2nd place** solution for the Computer Vision Hackathon [Game of Deep Learning](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/) organised by Analytics Vidhya.

## Problem Statement

Ship or vessel detection has a wide range of applications, in the areas of maritime safety, fisheries management, marine pollution, defence and maritime security, protection from piracy, illegal migration, etc.

Keeping this in mind, a Governmental Maritime and Coastguard Agency is planning to deploy a computer vision based automated system to identify ship type only from the images taken by the survey boats. You have been hired as a consultant to build an efficient model for this project.

## Dataset Description 

There are 6252 images in train and 2680 images in test data. The categories of ships and their corresponding codes in the dataset are as follows -

There are 5 classes of ships to be detected which are as follows: 

  * Cargo 
  * Military 
  * Carrier 
  * Cruise 
  * Tankers
## Evaluation Metric 

  Weighted F1 score

## Approach

- Each model is trained in 3 stages using progressive resizing : `128x128 -> 256*256 -> 299x299`

- Various combinations of techniques were used like `Training on FP16`, `Data Augmentations(flip left right, random zoom and crop, etc)`, `Mixup` with `Focal Loss` and `FlattenedLoss of CrossEntropyLoss`.

- Final Submission was generated using `Final Blending`  notebook. Used `Avg of predictions of 3 models` for final submission as it performed better on Public LB. 

| Model  | Public LB  Score| Private LB Score |  
|---|---|---|
| ResNet50 |0.98127|0.97129|
| ResNeXt50 |0.98320|0.97822| 
| SeResNeXt50 |0.98031|0.98066| 
|Avg of predictions of 3 models| 0.98599 | 0.98567 |
|Avg of TTA predictions of 3 models| 0.98410 | 0.98815 |

## LeaderBoard 

- [Public LB](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/lb) : **0.98599** & **6th out of 2083 participants**
- [Private LB](https://datahack.analyticsvidhya.com/contest/game-of-deep-learning/pvt_lb) : **0.98567** & **2nd out of 2083 participants**

## Setting up environment
```
fastai==1.0.52
pretrainedmodels==0.7.4
```
Models were trained on Colab using `Python 3` notebooks, so other necessary packages were already installed.

## Steps to Reproduce 

   * Extract `train.zip` in `data` folder and remove `_MACOSX` and `train.zip` file.
   * Run the notebooks `Final_ResNet50`, `Final_ResNeXt50` and `Final_SeResNeXt50`.
   * Run the `Final_Blending` notebook on the generated outputs from the three notebooks.
   
Also predicted probabilities of the three models are provided in `PredictedProbabilities` folder and the two submission files are provided in `FinalSubmission` folder
