
from tqdm.auto import tqdm
from timeit import default_timer as timer
from torch.utils.data import DataLoader
from food_vision_project.engine import train_model , test_model

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

  start_time = timer()

  for epoch in tqdm(range(epochs)):
    train_acc , train_loss = train_model(model , train_data , loss_fn , optimizer , device)

    test_acc , test_loss = test_model(model , test_data , loss_fn , device)

    test_loss_values.append(test_loss)
    test_accuracy_values.append(test_acc)
    train_loss_values.append(train_loss)
    train_accuracy_values.append(train_acc)

    print(f"Epoch {epoch} | Train Loss: {train_loss:0.4f} | Train Accuracy: {train_acc:0.2f} | Test Loss: {test_loss:0.4f} | Test Accuracy: {test_acc:0.2f} ")

  end_time = timer()
  process_time = end_time - start_time
  print(f"\nProcess Time: {process_time:0.2f} seconds")

  result_values["test_loss"] = test_loss_values
  result_values["test_accuracy"] = test_accuracy_values
  result_values["train_loss"] = train_loss_values
  result_values["train_accuracy"] = train_accuracy_values

  return result_values
