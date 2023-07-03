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
from torch.nn.modules.dropout import Dropout


try:
  import torchinfo
except:
  !pip install torchinfo
from torchinfo import summary



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

  fig , axs = plt.subplots(n_rows , n_columns , sharex = True , sharey = True , figsize = (10 , 10))

  for i in range(n_rows):
    for j in range(n_columns):
      axs[i , j].imshow(sample_image[i * patch_size: (i + 1) * patch_size , j * patch_size: (j + 1) * patch_size , :])
      axs[i , j].set_xlabel(j+1)
      axs[i , j].set_ylabel(i+1 , rotation  = "horizontal" , ha = "right")
      axs[i , j].set_xticks([])
      axs[i , j].set_yticks([])
      axs[i , j].label_outer()
  plt.show()

#show the result
image = torch.permute(images[0] , (1 , 2  , 0))
plot_patched_image(image , patch_size= 16)

create prepare method to make images in embedded form
def prepare_embed_input(image):

  '''prepare data embed for our transfomer in a apropriate shape

      input: image torch.utils.data.Dataloder

      output: array
  '''
  conv = torch.nn.Conv2d(in_channels = 3 , out_channels = 16 * 16 * 3 , kernel_size = 16 , stride = 16 , padding = 0)
  x = conv(torch.unsqueeze(torch.permute(image , (2 , 0 , 1)) , dim = 0))
  flat = torch.nn.Flatten(start_dim = 2 , end_dim = 3)
  return torch.permute(flat(x) , (0 , 2 , 1))



#plot some embedded visually
fig , axs = plt.subplots(3 , 3 , figsize = (5 , 5 ), sharex = True , sharey = True)
n_fig = 0

for i in range( 3):
  for j in range(3):
    axs[i , j].imshow(torch.reshape(torch.squeeze(torch.sigmoid(flatten_input) , dim = 0)[n_fig] , shape = (16 , 16 , 3)).detach().numpy())
    n_fig = n_fig + 1
    axs[i , j].set_xlabel(j + 1)
    axs[i , j].set_ylabel(i + 1 , rotation = "horizontal" , ha = "right")
    axs[i , j].set_xticks([])
    axs[i , j].set_yticks([])
    axs[i , j].label_outer()



# create Patching class befor adding class token and position embeddings
class PatchEmbedding(torch.nn.Module):


  def __init__(self ,
               in_heights = 224 ,
               in_widths = 224 ,
               in_channels = 3,
               patch_size = 16 ,
               dropout_mlp:float = 0.1):

    super().__init__();

    self.dropout = torch.nn.Dropout(p = dropout_mlp)

    self.class_token = torch.nn.Parameter(
        torch.rand(1 , 1 , patch_size * patch_size * in_channels),
        requires_grad = True
        )

    self.position_embeds = torch.nn.Parameter(
        torch.rand((1, int(in_heights * in_widths / patch_size**2) + 1, patch_size*patch_size*in_channels)),
        requires_grad = True)

    self.patches = torch.nn.Conv2d(in_channels = in_channels,
                                   out_channels = patch_size * patch_size * in_channels,
                                   kernel_size = patch_size,
                                   stride = patch_size,
                                   padding = 0)

    self.flat_patches = torch.nn.Flatten(start_dim = 2 , end_dim = 3)


  def forward(self , x):

    patches = self.patches(x)

    flat_patches = self.flat_patches(patches)
    #print(flat_patches.shape)
    flat_patches = torch.permute(flat_patches , (0 , 2 , 1))
    #print(flat_patches.shape)
    class_token = self.class_token.expand(x.shape[0] , -1 , -1)
    flat_patches_with_class_token = torch.cat((flat_patches , class_token) , dim = 1)
    #print(flat_patches_with_class_token.shape)
    position_embeds = self.position_embeds.expand(x.shape[0] , -1 , -1)
    flat_patches_with_class_token_and_position_embeds = flat_patches_with_class_token + position_embeds
    #dropout directly after adding positional- to patch embeddings
    flat_patches_with_class_token_and_position_embeds = self.dropout(flat_patches_with_class_token_and_position_embeds)
    #print(flat_patches_with_class_token_and_position_embeds.shape)
    return flat_patches_with_class_token_and_position_embeds



class MultiHeadAttention(torch.nn.Module):
  '''
    create part 2 of the model ViT

    inputs:
      embed_patch_size: int -> it is equal to dimension of your embedding
      num_heads: int -> how many different heads you need
      dropout: int -> 0 according to the article
    outputs:
      attn_output: it a tensor
  '''
  def __init__(self,
               embed_patch_size:int = 768,
               num_heads:int = 12,
               dropout_msa: int = 0):

    super().__init__();

    self.norm = torch.nn.LayerNorm(normalized_shape = embed_patch_size , device = device)

    self.multi_head_attn = torch.nn.MultiheadAttention(embed_dim = embed_patch_size,
                                                       num_heads = num_heads,
                                                       dropout = dropout_msa,
                                                       batch_first = True)

  def forward(self, x):

    x = self.norm(x)
    attn_output , _ = self.multi_head_attn(query = x , key = x , value = x , need_weights = False)
    return attn_output




class MultiLayerPerceptron(torch.nn.Module):
  '''
    MLP block of the ViT model
  '''

  def __init__(self , embed_patch_size:int = 768 , MLP_size:int = 3072 , dropout_mlp:float = 0.1):
    super().__init__();

    self.norm = torch.nn.LayerNorm(normalized_shape = embed_patch_size, device = device)

    self.multilayerperceptron = torch.nn.Sequential(
        torch.nn.Linear(in_features = embed_patch_size, out_features= MLP_size),
        torch.nn.GELU(),
        torch.nn.Dropout(p = dropout_mlp),
        torch.nn.Linear(in_features = MLP_size , out_features = embed_patch_size),

    )


  def forward(self, x):
    x = self.norm(x)
    x = self.multilayerperceptron(x)
    return x



class TransformerEncoder(torch.nn.Module):
  '''
    TransformerEncoder for Vit Model
  '''

  def __init__(self , embed_patch_size:int = 768,
               num_heads:int = 12,
               MLP_size:int = 3072,
               dropout_msa: int = 0,
               dropout_mlp:float = 0.1):

    super().__init__()

    self.msa = MultiHeadAttention(embed_patch_size = embed_patch_size , num_heads = num_heads , dropout_msa = dropout_msa)

    self.mlp = MultiLayerPerceptron(embed_patch_size = embed_patch_size , MLP_size = MLP_size, dropout_mlp = dropout_mlp)

  def forward(self, x):

    x = self.msa(x) + x

    x = self.mlp(x) + x

    return x


# create ViT Model using befor classes
class ViT(torch.nn.Module):
  '''create ViT model
  '''

  def __init__(self ,
               color_channel:int = 3,
               image_width:int = 224,
               image_height:int = 224,
               patch_size:int = 16,
               num_heads:int = 12,
               num_layers:int = 12,
               MLP_size:int = 3072,
               num_classes:int = 3,
               dropout_mlp:float = 0.1,
               dropout_msa:int = 0):

    super().__init__();

    assert image_width % patch_size == 0 , "Image Size is not devidable with Path_size, use another patch or Image size"

    self.patchify = PatchEmbedding(in_heights = image_height,
                                   in_widths = image_width,
                                   in_channels = color_channel,
                                   batch_size = batch_size,
                                   dropout_mlp = dropout_mlp)


    self.transformer = TransformerEncoder(embed_patch_size = patch_size*patch_size*color_channel,
                                          num_heads = num_heads,
                                          MLP_size = MLP_size,
                                          dropout_msa = dropout_msa,
                                          dropout_mlp = dropout_mlp)

    self.transformer_layers = torch.nn.Sequential(*[
        self.transformer for _ in range(num_layers)
    ])

    self.mlp_head = torch.nn.Sequential(
        torch.nn.LayerNorm(normalized_shape = patch_size*patch_size*color_channel),
        torch.nn.Linear(in_features = patch_size*patch_size*color_channel , out_features = num_classes)
    )



  def forward(self , x):
    embeds = self.patchify(x)
    transformers_output = self.transformer_layers(embeds)
    output = self.mlp_head(transformers_output[: , 0])
    return output





# lets test our first trial
vit_model = ViT()
vit_model.to(device)

optimizer = torch.optim.Adam(params = vit_model.parameters() , lr = 0.001)
loss_fn = torch.nn.CrossEntropyLoss()

results1 = train(vit_model,
                 epochs = 5 ,
                 train_data = train_dataLoader ,
                 test_data = test_dataLoader ,
                 loss_fn = loss_fn ,
                 optimizer = optimizer,
                 device = device)

# plot results1 train and test/train loss or test/train accuracy
plt.figure(figsize = (20 , 7))
plt.subplot(1 , 2 , 1)
plt.plot(results1["test_loss"])
plt.plot(results1["train_loss"])
plt.legend(["test_loss" , "train_loss"])
plt.subplot(1 , 2 , 2)
plt.plot(results1["test_accuracy"])
plt.plot(results1["train_accuracy"])
plt.legend(["test_accuracy" , "train_accuracy"])



from torchvision.models.vision_transformer import ViT_B_16_Weights , vit_b_16

vit_weights = ViT_B_16_Weights.DEFAULT
vit_model_pretrained = vit_b_16(weights = vit_weights)

for params in vit_model_pretrained.parameters():
  params.requires_grad = False

vit_transformer = vit_weights.transforms()
vit_model_pretrained.heads = torch.nn.Sequential(torch.nn.Linear(in_features = 768 , out_features = 3))

test_dataset_pretrained = datasets.ImageFolder(test_data_path , transform = vit_transformer)
train_dataset_pretrained = datasets.ImageFolder(train_data_path , transform = vit_transformer)

test_dataLoader_pretrained = DataLoader(test_dataset_pretrained,
                             batch_size = BATCH_SIZE,
                             shuffle = False,
                             num_workers = torch.cuda.device_count(),
                             drop_last=True) #drop_last will delete last batch if not in 32 size


train_dataLoader_pretrained = DataLoader(train_dataset_pretrained,
                              batch_size = BATCH_SIZE,
                              shuffle = True,
                              num_workers = torch.cuda.device_count(),
                              drop_last = True)

summary(vit_model_pretrained , (32 , 3 , 224 , 224) ,
        col_names = ["input_size" , "output_size" , "num_params" , "trainable"], 
        col_width = 20)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = vit_model_pretrained.parameters() , lr = 0.001)

#train the pretrained model with our data
results2 = train(vit_model_pretrained ,
                 epochs = 10 ,
                 train_data = train_dataLoader_pretrained ,
                 test_data = test_dataLoader_pretrained ,
                 loss_fn = loss_fn ,
                 optimizer = optimizer ,
                 device = device)


#plot loss and accuracy curves
plt.figure(figsize = (20 , 7))
plt.subplot(1 , 2 , 1)
plt.plot(results2["test_loss"])
plt.plot(results2["train_loss"])
plt.legend(["test_loss" , "train_loss"])
plt.subplot(1 , 2 , 2)
plt.plot(results2["test_accuracy"])
plt.plot(results2["train_accuracy"])
plt.legend(["test_accuracy" , "train_accuracy"])

#save our best model for further usage
torch.save(vit_model_pretrained.state_dict() , "/content/Pytorch/ViT_food_vision_project/saved_models/vit_model_pretrained.pth")
