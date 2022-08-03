# E2E Bahavior Classification

## Introduction

This project is the implementation of my master's thesis: **End-To-End System For Road Agents Behavior Classification Based On Dash Cam Image**, in the following will describe how this project work, what and how dataset is used.

<p align="center">
  <img src="./imgs/demo/BC.gif" alt="drawing" width="350" />
</p>

## Abstract
> The classification of road agents' behavior is an important study in the field of self-driving. Identifying which road agents are inclined to dangerous or normal behavior during the driving process can improve the safety of the driving process. In addition to the importance of road agent behavior classification in the field of self-driving, it can also achieve a certain degree of driving assistance to remind and advise the driver to reduce the risk of driving in the process. The system can use the results of multi-object tracking to form a historical trajectory, and by turning on trajectory prediction, the system can predict the future path, and then use the historical and predicted trajectory to classify individual road agents as aggressive or conservative drivers, and use overtaking assistance to provide users with driving recommendations under normal conditions. We also design a highly efficient parallelization system for the dependency and independence of the input and output of each module, so that the whole system can reach the speed of real-time detection.

## Installation

### Enviornment

- Python: 3.7
- Pytorch: 1.10.0
- CUDA: 10.2
- cuDNN: 8.0.3.33-1
- TensorRT: 7.1.3-1

How to install this Project:
```sh
git clone https://github.com/leisurecodog/E2E-Behavior-Classification.git
pip install requirements.txt
```

**The pipeline version is no longer maintained**, all of newest implementation is placed in **parallel** folder, if you need to run pipeline version, please follow the parallel version to modify.

How to run this project:
```sh
cd parallel
python system_main.py
```
And the UI will be displayed like below image:

<img src="./imgs/demo/UI.jpg" width="450" />

### Usage

- MOT: Multiple Object Tracking, modified from [ByteTrack](https://github.com/ifzhang/ByteTrack).

- TP: Trajectory Prediction, modified from [DDPG](https://github.com/ghliu/pytorch-ddpg).
- BC: Behavior Classification, modified from [GraphRQI](https://github.com/rohanchandra30/GraphRQI).
- OT: Overtaking Assistance, combined from [PINet](https://arxiv.org/abs/2002.06604), [Yolact_Edge](https://github.com/haotian-liu/yolact_edge) and our overtaking detection algorithm. Please follow YolactEdge to calibration TensorRT, you can also execute this project without calibration (follow [there](https://github.com/haotian-liu/yolact_edge#inference-without-calibration)).

If you want to modify the module, please follow the above abbreviation description to find corresponding folder.

If you want to replace the module, you need to rewrite the corresponding file in the module folder, such as TP_module/TP.py

## Train & Test
Each folder of module have a folder name call **source_code** except OT module, this folder contains all of original files. if needs any training, testing or others, you can use this folder.

In the following guide will describe how to use  source code of each module.

### MOT module

If you want to train original ByteTrack, please click the [github link](https://github.com/ifzhang/ByteTrack) and follow the guide.

Or you can just train other object detector like [SSD](https://arxiv.org/abs/1512.02325?context=cs) or others, then replace the YOLOX to what you train like [demo_track_yolov5.py](./parallel/MOT_module/source_code/tools/demo_track_yolov5.py).

Evaluate and Test also can follow original ByteTrack guide.

### TP module

If you want to training this method, you need to prepare data like [this](#bdd100kbdd100k-mot), then modify code in 

## Dataset
### BDD100K/BDD100K MOT

BDD100K: [link](https://www.bdd100k.com/)<br>
This dataset is used to Training Multiple Object Tracking, Trajectory Prediction and Behavior Classification.<br>
<!-- If you need to training MOT module(ByteTrack), you can just follow other model that how to train a
object detector, then replace the detector in ByteTrack and modify corresponding code. <br> -->
For traning the Trajectory Prediction, you need to follow the above format:
```sh
# w: image weight
# h: image height
# format: object_class object_id center_x/w center_y/h object_w/w object_h/h 
0 20 0.15 0.2 0.03 0.01 # example
```
and make sure that your images and labels is placed in corresponding location and name, like:
```sh
bdd100k/images/track/train/00a0f008/00a0f008-00001.jpg # image location, for get image width and height
bdd100k/labels_with_ids/track/train/00a0f008/00a0f008-00001.txt # annotation location
```

For training the Behavior Classification, you need to label data by self, bolow is the example of label file for a video:
```sh
# label format: object_id behavior_label.
# 0 means conservative, 1 means aggressive.
1 0 
2 1
...
```

### KITTI/KITTI Tracking
KITTI: [link](http://www.cvlibs.net/datasets/kitti/)<br>
This dataset is used to Evaluate Trajectory Prediction and Overtaking Assistance.<br>
You can just downaload the KITTI Tracking dataset if you just want to evaluate Trajectory Prediction.<br>
If you want to evaluate overtaking assistance, you need to download the raw data and LiDAR data of KITTI, and annotate overtaking data by yourself.
### CEO Dash Cam Videos
This dataset is used to demo our total result, neither training nor testing. Due to the copyright problem, we can not release this dataset, you can use other dash cam videos to demo.