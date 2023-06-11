import torch
import torchvision
import scipy
import numpy
import sklearn
import pandas
import matplotlib.pyplot as plt
import os
import timeit
import random


from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

#load data
test_data_path = "/content/Pytorch/ViT_food_vision_project/pizza_steak_sushi/test"
train_data_path = "/content/Pytorch/ViT_food_vision_project/pizza_steak_sushi/train"

#set image size and make transformer
IM_SIZE = 224
BATCH_SIZE = 32

transformer = transforms.Compose([
    transforms.Resize((IM_SIZE , IM_SIZE)),
    transforms.ToTensor()
])

#create test and train dataLoaders
test_dataset = datasets.ImageFolder(test_data_path , transform = transformer)
train_dataset = datasets.ImageFolder(train_data_path , transform = transformer)

class_names = train_dataset.classes

test_dataLoader = DataLoader(test_dataset, batch_size = BATCH_SIZE , shuffle = False , num_workers = torch.cuda.device_count())
train_dataLoader = DataLoader(train_dataset, batch_size = BATCH_SIZE , shuffle = True , num_workers = torch.cuda.device_count())

#show random images of data
columns = 3
rows = 4

data = next(iter(train_dataLoader))

rand = random.randint(0 , len(data[0]) - (columns * rows))

images = data[0][rand:rand + (columns * rows)]
targets = data[1][rand:rand + (columns * rows)]

plt.figure(figsize = (10 , 7))
for i in range(0 , (columns * rows)):
  plt.subplot(columns , rows , i + 1)
  plt.imshow(torch.permute(images[i] , dims = (1 , 2 , 0)))
  plt.title(class_names[targets[i]])
  plt.axis(False)


#create a function to plot an image after empatching it
def plot_patched_image(image , patch_size:int):
  ''' 
    plot an image after cropping image into patchs
      
      inputs:
        image: an torch permutetd image in shape (H , W , C)
        patch_size : an integere odd number


      output:
        plot a figure with patched sub images
  '''

  patch_size = patch_size
  sample_image = image
  number_of_patches = int(sample_image.shape[0] / patch_size)
  n_rows = number_of_patches
  n_columns = n_rows
  n_fig = 1

  plt.figure(figsize = (patch_size , patch_size))
  for i in range(n_rows):
    for j in range(n_columns):
      plt.subplot(n_rows , n_columns , n_fig)
      plt.imshow(sample_image[i * patch_size: (i + 1) * patch_size , j * patch_size: (j + 1) * patch_size , :])
      plt.axis(False)
      n_fig = n_fig + 1

#show the result
image = torch.permute(images[0] , (1 , 2  , 0))
plot_patched_image(image , patch_size= 16)
