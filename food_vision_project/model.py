import torch

Conv_Kernel_Size = 3
Conv_Stride_Size = 1
Conv_Padding_Size = 0

Max_Kernel_Size = 2
Max_Stride_Size = 1

class TinyVGG(torch.nn.Module):
  ''' 
    Tiny VGG Network created from https://poloclub.github.io/cnn-explainer/
  '''

  def __init__(self , input_size:int , hidden_units:int , output_shape:int):
    super().__init__();
    self.conv_block1 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = input_size , out_channels = hidden_units,
                        kernel_size = Conv_Kernel_Size ,stride = Conv_Stride_Size,
                        padding = Conv_Padding_Size),

        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels = hidden_units , out_channels = hidden_units,
                        kernel_size = Conv_Kernel_Size ,stride = Conv_Stride_Size,
                        padding = Conv_Padding_Size),
        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size = Max_Kernel_Size ,stride = Max_Stride_Size)
    )

    self.conv_block2 = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels = hidden_units , out_channels = hidden_units,
                        kernel_size = Conv_Kernel_Size , stride = Conv_Stride_Size , padding = Conv_Padding_Size),

        torch.nn.ReLU(),

        torch.nn.Conv2d(in_channels = hidden_units , out_channels = hidden_units,
                        kernel_size = Conv_Kernel_Size , stride = Conv_Stride_Size , padding = Conv_Padding_Size),

        torch.nn.ReLU(),

        torch.nn.MaxPool2d(kernel_size = Max_Kernel_Size , stride  = Max_Stride_Size)
    )

    self.last_layer = torch.nn.Sequential(
        torch.nn.Flatten(),

        torch.nn.Linear(in_features = hidden_units * 54 * 54, out_features=output_shape)
    )


  def forward(self , x):
    x = self.conv_block1(x)
    #print(x.shape)
    x = self.conv_block2(x)
    #print(x.shape)
    x = self.last_layer(x)
    #print(x.shape)
    return x
    
