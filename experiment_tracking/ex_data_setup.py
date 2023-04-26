
import torch
import torchvision

from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path

try:
  import torchinfo
except:
  !pip install torchinfo

import numpy
import scipy
import matplotlib.pyplot as plt
import pandas
import os

import data_setup, engine, model, predict, train


device = "cuda" if torch.cuda.is_available else "cpu"

data_dir = "/content/Pytorch/experiment_tracking/pizza_steak_sushi"
test_dir = data_dir + "/test"
train_dir = data_dir + "/train"

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

                                 transformer = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224 , 224)),
    torchvision.transforms.ToTensor(),
    normalize
])

train_data = datasets.ImageFolder(train_dir , transform = transformer)
test_data = datasets.ImageFolder(test_dir , transform = transformer)

train_dataLoader = DataLoader(train_data , batch_size = 32 , shuffle = True , num_workers = torch.cuda.device_count())
test_dataLoader = DataLoader(test_data , batch_size = 32 , shuffle = False , num_workers = torch.cuda.device_count())

class_names = train_data.classes
class_names_idx = train_data.class_to_idx
