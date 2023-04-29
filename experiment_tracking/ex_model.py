
import torch
import torchvision

from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path

try:
  import torchinfo
except:
  !pip install torchinfo
from torchinfo import summary

import numpy
import scipy
import matplotlib.pyplot as plt
import pandas
import os

import data_setup, engine, model, predict, train




# load efficient b0 weights
efficient_w = torchvision.models.EfficientNet_B0_Weights.DEFAULT

# initiate efficent model with b0 weights
efficient_model = torchvision.models.efficientnet_b0(weights = efficient_w)

# change last layer of the network with our data output
classifier = torch.nn.Sequential(
    torch.nn.Dropout(p = 0.2 , inplace = True),
    torch.nn.Linear(in_features = 1280 , out_features = len(class_names))
)
efficient_model.classifier = classifier

# freeze all features layers for just training last layers
for params in efficient_model.features.parameters():
  params.requires_grad = False

# see how out network look like
summary(efficient_model , input_size = (1 , 3, 224 , 224),
        col_names=["input_size" , "output_size" , "num_params" , "trainable"] , verbose = 0)
