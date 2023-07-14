import torch
from torch.utils.data import DataLoader

from torchvision.datasets import ImageFolder
from torchvision import transforms

import os
from pathlib import Path


def create_dataLoader(train_dir:Path,
                      test_dir:Path,
                      batch_size:int,
                      transformer:transforms.Compose):
  '''
    Make test/train data Loder object

    inputs:
      train_dir: train data directory
      test_dir: test data directory
      batch_size: batch size of the dataLoader
      transformer: transforms.Compose object 
    
    return:
      train_dataLoader: torch.utils.data.DataLoader object
      test_dataLoader: torch.utils.data.DataLoader object
      class_names: list of the all class names
  '''

  BATCH_SIZE = batch_size
  TRAIN_DIR = train_dir
  TEST_DIR = test_dir
  NUM_WORKERS = os.cpu_count()

  TRANSFORMER = transformer



  train_dataset = ImageFolder(TRAIN_DIR , transform = TRANSFORMER)
  test_dataset = ImageFolder(TEST_DIR , transform = TRANSFORMER)


  train_dataLoader = DataLoader(train_dataset ,
                                batch_size = BATCH_SIZE,
                                shuffle = True,
                                num_workers = NUM_WORKERS,
                                pin_memory = True)

  test_dataLoader = DataLoader(test_dataset,
                              batch_size = BATCH_SIZE,
                              shuffle = False,
                              num_workers = NUM_WORKERS,
                              pin_memory=True)
  
  class_names = train_dataset.classes
  

  return train_dataLoader , test_dataLoader , class_names
