from tqdm.auto import tqdm
import timeit
from helper_functions import accuracy_fn
import torch
from matplotlib.pyplot import plt

device = "cuda" if torch.cuda.is_available() else "cpu"

def calclulate_timer(start_time , end_time , device = device):
  ''' this method's been created for calculating process time on the selcted device
  args:
     start_time: timeit.Timer().timer() before operation
     end_time: timeit.Timer().timer() after operation
  output:
     print end_time - start_time 
  '''
  start = start_time
  end = end_time
  time = end - start
  return time



#create evaluation method
def model_evaluation(model:torch.nn.Module, loss_function:torch.nn.Module, dataLoader:torch.utils.data.DataLoader , device = device):
  ''' this method will evaluate the model
  arg:
     model: your model which you want to evaluate
     loss_function: your loss function
     dataLoader: your data that in the format of torch.utils.data.DataLoader
     device: which device you are runing the code
  output: return a dicianary contains
     test loss
     test_accuracy
     process time
     device
     '''
     
  start = timeit.Timer().timer()
  device = "cuda" if torch.cuda.is_available() else "cpu"
 
  model.eval()
  with torch.inference_mode():
    
    test_loss = 0
    test_accuracy = 0

    for data , label in dataLoader:

      data = data.to(device)
      label = label.to(device)
      preds = model(data)

      loss = loss_function(preds , label)
      test_loss = test_loss + loss

      accuracy = accuracy_fn(label , torch.argmax(preds, dim = 1))
      test_accuracy = test_accuracy + accuracy

    
    test_loss = test_loss / len(dataLoader)
    test_accuracy = test_accuracy / len(dataLoader)
  
  end = timeit.Timer().timer()
  evaluation_process_time = calclulate_timer(start_time = start , end_time = end , device = device)

  return {"Test Loss" : round(test_loss.item() , 4),
          "Test Accuracy": round(test_accuracy , 2),
          "Evaluation Time":f"{evaluation_process_time:0.2f}s",
          "Model Name":f"{model.__class__.__name__}",
          "Evaluated On":device}



def training(model:torch.nn.Module,
             dataLoader:torch.utils.data.DataLoader,
             loss_function:torch.nn.Module,
             optimizer:torch.optim.Optimizer,
             accuracy_function,
             device = device):
  
  model.train()
  train_loss = 0
  train_accuracy = 0

  for batch , (data , label) in enumerate(dataLoader):

    data = data.to(device)
    label = label.to(device)

    train_preds_logits = model(data)
    train_preds = torch.argmax(train_preds_logits , dim = 1)
    accuracy = accuracy_function(label , train_preds)

    loss = loss_function(train_preds_logits , label)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if batch % 400 == 0:
      print(f"number of data passed {batch * len(data)}/{len(dataLoader) * len(data)}")
    
    train_loss = train_loss + loss
    train_accuracy = train_accuracy + accuracy
  
  train_loss = train_loss / len(dataLoader)
  train_accuracy = train_accuracy / len(dataLoader)

  print(f"Train Loss: {train_loss:0.5f} | Train Accuracy: {train_accuracy:0.2f}%")


def testing(model:torch.nn.Module,
             dataLoader:torch.utils.data.DataLoader,
             loss_function:torch.nn.Module,
             accuracy_function,
             device = device):
  
  test_loss = 0
  test_accuracy = 0

  model.eval()
  with torch.inference_mode():
    for batch , (data , label) in enumerate(dataLoader):
      data = data.to(device)
      label = label.to(device)

      test_preds_logits = model(data)
      test_preds = torch.argmax(test_preds_logits , dim = 1)
      accuracy = accuracy_function(label , test_preds)

      loss = loss_function(test_preds_logits , label)

      test_loss = test_loss + loss
      test_accuracy = test_accuracy + accuracy
    
  test_loss = test_loss / len(dataLoader)
  test_accuracy = test_accuracy / len(dataLoader)

  print(f"Test Loss: {test_loss:0.5f} | Test Accuracy: {test_accuracy:0.2f}%\n")

import random
def plot_random_predictions(model:torch.nn.Module, class_labels:list, data, rows = 3 , columns = 3,
 figure_size = (15 , 10)):
  '''plot random data with their predicts '''
  samples = []
  labels = []

  plt.figure(figsize = figure_size)

  for i in range(rows * columns):
    rand = random.randint(0 , len(data))
    sample , label = data[rand]
    samples.append(sample)
    labels.append(label)

    plt.subplot(rows , columns , i + 1)
    plt.imshow(sample.squeeze() , cmap = "gray")

    pred_logits = model(sample.unsqueeze(dim = 0).to(device))
    pred = torch.argmax(pred_logits , dim = 1)
    pred = class_labels[pred]

    title = f"Predict: {pred} | Label: {class_labels[label]}"

    if pred == class_labels[label]:
      plt.title(title , fontsize = 12 , color = "green")
    else:
      plt.title(title , fontsize = 12 , color = "red")
    plt.axis(False)
