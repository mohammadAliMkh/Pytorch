
import matplotlib.pyplot as plt
import torch
import random
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms

def show_random_predict_images(model:torch.nn.Module,
                   data:torch.utils.data.DataLoader,
                   class_names:list,
                   number_rows:int = 3,
                   number_columns:int = 3,
                   figsize:tuple = (10 , 7)):
  '''
    plot a rows * columns figure include some images with thier predicted labels

      args:
          model: torch.nn.Module,
          data: torch.utils.data.DataLoader
          class_names: list of the all class names
          number_rows: number of rows (default = 3, integer)
          number_columns: number of columns (default = 3 , integer)
          figsize: tuple (default = (10 , 7))

      returns:
          plot a plt with figsize and predict labels
          if predict correct (label -> green)
          if predict wrong (abel -> red)
  '''

  print("There exists only 11 images in the last batch,")
  print("Please Consider this issues and set the rows and columns under 3.\n")

  plt.figure(figsize = figsize)

  list_imgs = []
  list_labels = []
  rand = random.randint(0 ,len(data)-1)

  for i , (X , y) in enumerate(data):
    list_imgs.append(X)
    list_labels.append(y)
  img , label = list_imgs[rand] , list_labels[rand]


  for i in range(number_rows * number_columns):
    plt.subplot(number_rows , number_columns , i + 1)
    plt.imshow(img[i].permute(1 , 2 , 0))
    predict = class_names[torch.argmax(model(torch.unsqueeze(img[i] , dim = 0)) , dim = 1)]  
    plt.axis(False)
    if (predict == class_names[label[i]]):
      plt.title(predict , c = "green")
    else:
      plt.title(predict , c = "red")


def predict_image_from_wild(image_url:str, model:torch.nn.Module, transformer:transforms, class_names:list):
  '''
    Get an Image from Wild and Make it to Tensor and Predict

      args:
          image_url: url string of the image
          model: torch.nn.Module
          transfomer: torchvision.transforms to reshape and make to tensor
          class_names: list of the all class names
      
      returns:
          plot the image and show predicted label on top of that
  '''
  response = requests.get(image_url)
  img = Image.open(BytesIO(response.content))

  img_tensor = transformer(img)
  predict = class_names[torch.argmax(model(torch.unsqueeze(img_tensor , dim = 0)) , dim = 1)]

  plt.figure(figsize = (10 , 7))
  plt.imshow(img)
  plt.axis(False)
  plt.title(predict)
