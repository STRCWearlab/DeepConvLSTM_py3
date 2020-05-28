# DeepConvLSTM_py3
DeepConvLSTM implemented in python 3 and pytorch

# Parameters

We can view the parameters for each layer by calling

```
for param in net.parameters():
  :     print(type(param),param.size())
```

which will give the output below.

```
<class 'torch.nn.parameter.Parameter'> torch.Size([64, 67, 5, 1])
<class 'torch.nn.parameter.Parameter'> torch.Size([64])
<class 'torch.nn.parameter.Parameter'> torch.Size([64, 64, 5, 1])
<class 'torch.nn.parameter.Parameter'> torch.Size([64])
<class 'torch.nn.parameter.Parameter'> torch.Size([64, 64, 5, 1])
<class 'torch.nn.parameter.Parameter'> torch.Size([64])
<class 'torch.nn.parameter.Parameter'> torch.Size([64, 64, 5, 1])
<class 'torch.nn.parameter.Parameter'> torch.Size([64])
<class 'torch.nn.parameter.Parameter'> torch.Size([512, 64])
<class 'torch.nn.parameter.Parameter'> torch.Size([512, 128])
<class 'torch.nn.parameter.Parameter'> torch.Size([512])
<class 'torch.nn.parameter.Parameter'> torch.Size([512])
<class 'torch.nn.parameter.Parameter'> torch.Size([512, 128])
<class 'torch.nn.parameter.Parameter'> torch.Size([512, 128])
<class 'torch.nn.parameter.Parameter'> torch.Size([512])
<class 'torch.nn.parameter.Parameter'> torch.Size([512])
<class 'torch.nn.parameter.Parameter'> torch.Size([18, 128])
<class 'torch.nn.parameter.Parameter'> torch.Size([18])
```

# Convolutional network
Lines 1-8 of this output show the parameters of the convolutional network. Each layer consists of a set of learnable weights in the shape *(out_channels,in_channels,kernel_size[0],kernel_size[1])*, and a set of biases of shape *(out_channels)*.

# LSTM network
The parameters of the lstm networks are shown in lines 9-16 of the output. Line 9 `<class 'torch.nn.parameter.Parameter'> torch.Size([512, 64])` describes the input-hidden weights of the first layer of the lstm, of shape *(4\*hidden_size,input_size)* which map the outputs from the last convolutional layer of 64 filters onto the hidden units of the lstm. These represent the input, forget, cell state and output weights, 128 of each. The next three lines are the hidden-hidden weights, the input-hidden bias and the hidden-hidden bias. Lines 13-16 describe the parameters of the second lstm layer, which takes the 128 outputs from the first layer as input. 

# Linear output layer
The last two lines of the output above describe the parameters of the final linear perceptron output layer, which weights and combines the output from the LSTM layers and maps them to the output vector space, which in this case is the 18 activity classes in the Opportunity gesture recognition dataset. 

