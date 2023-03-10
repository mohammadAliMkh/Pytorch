import torchvision
import torch
import matplotlib.pyplot as plt
import os
import shutil
import torchinfo

import data_setup

from pathlib import Path
from torchvision import transforms

#get efficient weights
efficient_weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT
efficient_transformer = efficient_weight.transforms()

#define train and test data directory
TRAIN_IMAGE_DIR = "/content/food_vision_project/pizza_steak_sushi/train"
TEST_IMAGE_DIR = "/content/food_vision_project/pizza_steak_sushi/test"

#make train , test data loaders using our last data_setup.py file
train_dataLoader , test_dataLoader , class_names = data_setup.create_dataLoader(TRAIN_IMAGE_DIR,
                             TEST_IMAGE_DIR,
                             batch_size = 32 , transformer = efficient_transformer)

#initiate efficient model 
efficient_model = torchvision.models.efficientnet_b0(weights = efficient_weight)

#show the model architecture
torchinfo.summary(efficient_model,
                  input_size = (1 , 3  ,224 , 224),
                  col_names = ["input_size" , "output_size", "num_params" , "trainable"],
                  col_width = 20 )
