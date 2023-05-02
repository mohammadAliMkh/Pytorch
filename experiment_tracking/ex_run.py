
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

import numpy as np
import scipy
import matplotlib.pyplot as plt
import pandas
import os

from datetime import datetime
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data_setup, engine, model, predict, train
from engine import train_model , test_model

# create a method to create wirter for tensorford using our paramaters
def create_writer(exp_name:str, model_name:str, extra:str = None):

  date = datetime.now().strftime("%Y-%m-%d")

  if extra:
    log_dir = os.path.join("/content/Pytorch/experiment_tracking/runs", exp_name, model_name, extra)
  else:
    log_dir = os.path.join("/content/Pytorch/experiment_tracking/runs", exp_name, model_name)

  print(f"[INFO] your summary writer {log_dir} created.")
  return SummaryWriter(log_dir = log_dir)

# change train def and add writer function to it
def train(model:torch.nn.Module,
          epochs:int,
          train_data:torch.utils.data.DataLoader,
          test_data:torch.utils.data.DataLoader,
          loss_fn:torch.nn.Module,
          optimizer:torch.optim.Optimizer,
          device = device):
  '''
    Train Model with evaluation in Epoch Times you select

    args:
        model: torch.nn.Module
        epochs: number of epochs you want to train your model
        train_data: torch.utils.data.DataLoader (train data)
        test_data: torch.utils.data.DataLoader (test data)
        loss_fn: torch.nn.Module
        optimizer: torch.optim.Optimizer
        device: device agnostic parameter (cuda or cpu)
    
    returns:
        result_values: dictionary of the all test and train values (loss and accuray)
  '''

  
  result_values = {"test_loss":[],
                   "train_loss":[],
                   "test_accuracy":[],
                   "train_accuracy":[]}

  test_loss_values = []
  test_accuracy_values = []
  train_loss_values = []
  train_accuracy_values = []

  writer = create_writer(exp_name = "Pizza_Steak_Sushi" , model_name = "efficient" , extra = "5_Epoch")

  start_time = timer()

  for epoch in tqdm(range(epochs)):
    train_acc , train_loss = train_model(model , train_data , loss_fn , optimizer , device)

    test_acc , test_loss = test_model(model , test_data , loss_fn , device)

    test_loss_values.append(test_loss)
    test_accuracy_values.append(test_acc)
    train_loss_values.append(train_loss)
    train_accuracy_values.append(train_acc)

    writer.add_scalars(main_tag = "Loss",
                       tag_scalar_dict = {"Train":train_loss , "Test":test_loss},
                       global_step = epoch)
    writer.add_scalars(main_tag = "Accuracy",
                       tag_scalar_dict = {"Train":train_acc, "Test":test_acc},
                       global_step = epoch)
    writer.add_graph(model = model , input_to_model = torch.randn(32, 3 , 224 , 224).to(device))

    print(f"Epoch {epoch} | Train Loss: {train_loss:0.4f} | Train Accuracy: {train_acc:0.2f} | Test Loss: {test_loss:0.4f} | Test Accuracy: {test_acc:0.2f} ")

  end_time = timer()
  process_time = end_time - start_time
  print(f"\nProcess Time: {process_time:0.2f} seconds")

  
  writer.close()

  result_values["test_loss"] = test_loss_values
  result_values["test_accuracy"] = test_accuracy_values
  result_values["train_loss"] = train_loss_values
  result_values["train_accuracy"] = train_accuracy_values

  return result_values

# colab commands to run tensorboard windows into the colab notebook
%load_ext tensorboard
%tensorboard --logdir /content/Pytorch/experiment_tracking/runs

# if you want to check your model outside of the notebook you can use below code to provide
# you a link and then click it and use tensorboard dev windows to work with tensorboard
!tensorboard dev upload --logdir runs --name "test expriment" --description "working for learning"
