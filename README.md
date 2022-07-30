# E2E Bahavior Classification
This project is the implementation of my thesis: End-To-End System For Road Agents Behavior Classification Based On Dash Cam Image, in the following will describe how this project work, what and how data is used.

## Abstract
> The classification of road agents' behavior is an important study in the field of self-driving. Identifying which road agents are inclined to dangerous or normal behavior during the driving process can improve the safety of the driving process. In addition to the importance of road agent behavior classification in the field of self-driving, it can also achieve a certain degree of driving assistance to remind and advise the driver to reduce the risk of driving in the process. The system can use the results of multi-object tracking to form a historical trajectory, and by turning on trajectory prediction, the system can predict the future path, and then use the historical and predicted trajectory to classify individual road agents as aggressive or conservative drivers, and use overtaking assistance to provide users with driving recommendations under normal conditions. We also design a highly efficient parallelization system for the dependency and independence of the input and output of each module, so that the whole system can reach the speed of real-time detection.

## Install

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