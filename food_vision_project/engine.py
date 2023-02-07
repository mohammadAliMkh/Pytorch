
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model:torch.nn.Module,
                data:torch.utils.data.DataLoader,
                loss_fn:torch.nn.Module,
                optimizer:torch.optim.Optimizer,
                device = device):
  '''
    Train Model on Each Epoch and Return Train Accuracy and Loss

    args: 
        model: torch.nn.Module
        data: torch.utils.data.DataLoader (train data)
        loss_fn: torch.nn.Module
        optimizer: torch.optim
        device: device agnostic parameter (cpu or cuda)

    outputs:
        acc: List of accuracies in each batch trained
        loss: List of losses in each batch traind
  '''
  
  accuracy = []
  loss = []

  model.train()

  for batch , (X , y) in enumerate(data):

    X , y = X.to(device) , y.to(device)

    train_logits = model(X)

    train_loss = loss_fn(train_logits , y)
    loss.append(train_loss.item())

    optimizer.zero_grad()

    train_loss.backward()

    optimizer.step()

    train_predicts = torch.argmax(torch.softmax(train_logits , dim = 1) , dim = 1)
    train_accuracy = sum(train_predicts == y for train_predicts , y in zip(train_predicts , y))/len(y)
    accuracy.append(train_accuracy)
  
  acc = sum(accuracy)/len(data)
  loss = sum(loss)/len(data)

  return acc.item()*100 , loss


def test_model(model:torch.nn.Module,
               data:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               device = device):
  '''
    Test Model on Each Batch and Return Test Accuracy/Loss

    args:
        model: torch.nn.Module
        data: torch.utils.data.DataLoader (test data)
        loss_fn: torch.nn.Module
        device: device agnostic parameter (cpu or cuda)

    outputs:
        acc: List of test accuracies in each test batch
        loss: List of test losses in each test batch
  '''
  
  accuracy = []
  loss = []

  model.eval()

  with torch.inference_mode():
    for batch , (X , y) in enumerate(data):
      X , y = X.to(device) , y.to(device)

      test_logits = model(X)

      test_loss = loss_fn(test_logits , y)
      loss.append(test_loss)

      test_predicts = torch.argmax(torch.softmax(test_logits , dim = 1) , dim = 1)
      test_accuracy = sum(test_predicts == y for test_predicts , y in zip(test_predicts , y)) / len(y)
      accuracy.append(test_accuracy)
    
    acc = sum(accuracy) / len(data)
    loss = sum(loss) / len(data)

    return acc.item() * 100 , loss.item()
