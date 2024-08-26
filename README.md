# Develop a deep learning pipeline (model + code infrastructure) for instance segmentation

For this task, I used a pretrained model with configuration COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x in the Detectron2 framework, for Mask R-CNN with ResNet-50 FPN backbone trained on COCO Instance Segmentation dataset.

## Model Overview

- **Mask R-CNN Architecture**: it uses Mask R-CNN architecture, which is designed for instance segmentation tasks.
- **ResNet-50 Backbone**: it uses ResNet-50 backbone, known for its effectiveness in extracting features from images.
- **FPN (Feature Pyramid Network)**: FPN enhances the ability to detect objects at various scales by creating a pyramid of feature maps at different resolutions.
- **Training Schedule (3x)**: it was trained three times longer than the standard one defined in Detectron2.

## Why Choose This Model?

- **High Performance**: Models trained with a 3x schedule generally achieve higher accuracy compared to those trained with shorter schedules.
- **Versatility**: Mask R-CNN models with ResNet-50 FPN are versatile and capable of handling a wide range of instance segmentation tasks.
- **Pre-trained Weights**: The availability of pre-trained weights allows for faster development cycles, as these models can be fine-tuned on specific datasets without training from scratch.
- **Community Support**: Being part of the Detectron2 ecosystem ensures robust community support, extensive documentation, and compatibility with a wide range of hardware and software configurations.

## How the code works

The implementation was done using google colab to train the chosen detectron2 model on a new dataset. Steps taken are as follows:

- Installation of detectron2 and its dependencies
-	Some basic setup like setting up detectron2 logger, importing some common libraries and importing some common detectron2 utilities
-	**Data Preparation**: register the custom dataset in coco format to detectron2 using the register_coco_instances() method. 
-	**Train on a custom dataset**: the COCO-pretrained model is fine-tune on the custom dataset:
  - Batch size:  2
  - Learning rate:  0.00025
  - Max. Iterations:  500
  - Number of classes:  14

  The training metrices was logged in a json file metrics.json and can also be seen the colab file. Below and also in the images directory are some results of the inferencing:

**Inference**: Running inference with the trained model using the Detectron2 framework.
![Image1](https://github.com/KcLimitless/instance_seg_demo/blob/master/images/person_dog1.PNG)

![Image2](https://github.com/KcLimitless/instance_seg_demo/blob/master/images/person_mask.PNG)

![Image3](https://github.com/KcLimitless/instance_seg_demo/blob/master/images/dog1.PNG)

![Image4](https://github.com/KcLimitless/instance_seg_demo/blob/master/images/fruits1.PNG)

**Evaluation using Average Precision (AP)**: I used the same training data for evaluation because the dataset I used does not have a validation data which is not ideal. This was just for demonstration purpose. The result of the evaluation was saved in evaluation.json file and can also be seen in the colab file.

**Evaluation results for bbox**: 

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 36.214 | 57.993 | 40.745 | 13.835 | 29.043 | 48.516 |

**Per-category bbox AP**: 

| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| Apple      | 0.000  | Car        | 47.306 | Cherry     | 28.511 |
| Chilly     | 0.000  | Cow        | 57.723 | Dog        | 60.644 |
| Grapes     | 30.135 | Helmet     | 49.094 | Koala      | 38.663 |
| Mask       | 60.000 | Person     | 62.494 | Person     | nan    |
| Person     | nan    | Strawberry | 0.000  |            |        |

**Evaluation results for segm**: 

|   AP   |  AP50  |  AP75  |  APs   |  APm   |  APl   |
|:------:|:------:|:------:|:------:|:------:|:------:|
| 39.619 | 57.208 | 43.480 | 17.289 | 32.201 | 55.311 |

**Per-category segm AP**: 

| category   | AP     | category   | AP     | category   | AP     |
|:-----------|:-------|:-----------|:-------|:-----------|:-------|
| Apple      | 0.000  | Car        | 65.733 | Cherry     | 34.205 |
| Chilly     | 0.000  | Cow        | 62.772 | Dog        | 74.228 |
| Grapes     | 33.858 | Helmet     | 43.343 | Koala      | 41.139 |
| Mask       | 63.784 | Person     | 56.370 | Person     | nan    |
| Person     | nan    | Strawberry | 0.000  |            |        |

**Model conversion to ONNX format**: using a module provided by detectron2, I was able to convert the model to ONNx format.

## Conclusion: 

Given that the same set of data was used for training, evaluation and inference which is not ideal but just for demonstration purpose, the model performed well given the number of iterations used for the training. But it still needs to be tested on unseen data to see how well it generalizes and possibly further fine-tune it to improve its performance on unseen data. 
