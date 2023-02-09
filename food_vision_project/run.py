
import argparse
import torch
import numpy as np
from torchvision import transforms

import data_setup , model , engine , train , predict


parser = argparse.ArgumentParser("Train TinyVgg argument parsers")

parser.add_argument("--TRD" , default = "/content/Pytorch/food_vision_project/pizza_steak_sushi/train" , type = str , help = "train directory path")
parser.add_argument("--TED" , default = "/content/Pytorch/food_vision_project/pizza_steak_sushi/test" , type = str , help = "test directory path")
parser.add_argument("--epochs" , default = 3 , type = int , help = "number of epochs (default = 3)")
parser.add_argument("--lr" , default = 0.001 , type = float , help = "learning rate (default = 0.001)")
parser.add_argument("--hu" , default = 10 , type = int , help = "number of hidden units (default = 10)")
parser.add_argument("--bs", default = 32 , type = int , choices=[16 , 32 , 64] , help = "batch size (default = 32, choices [16 , 32 , 64])")

opt = parser.parse_args()

train_dir = opt.TRD
test_dir = opt.TED

device = "cuda" if torch.cuda.is_available() else "cpu"

transformer = transforms.Compose([
    transforms.Resize(size = (64 , 64)),
    transforms.ToTensor()
])

train_data , test_data , class_names = data_setup.create_dataLoader(train_dir , test_dir , batch_size = opt.bs, transformer = transformer)

tinyVgg = model.TinyVGG(input_size = 3 , hidden_units = opt.hu , output_shape = 3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(tinyVgg.parameters() , lr = opt.lr)

results = train.train(tinyVgg , epochs = opt.epochs ,
                      train_data = train_data,
                      test_data = test_data,
                      loss_fn = loss_fn,
                      optimizer = optimizer,
                      device = device)


print("Train Loss:" , np.mean(results["train_loss"]))
print("Train Accurac:" , np.mean(results["train_accuracy"]))
print("Test Loss:" , np.mean(results["test_loss"]))
print("Test Accuracy:" ,np.mean(results["test_accuracy"]))
