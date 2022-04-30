import torch
from typing import Union
import ConvCUDA

class MyNetwork:
    '''Python API for CUDA networks
    
    parameters
    -----------------
    inC : int - Number of input channels
    outC : int - Number of output channels
    hidC : int - Number of hidden channels
    WH : int - kernel size
    weights : torch.Tensor - Weights of the convolution
    bias : torch.Tensor - Bias of the convolution
    
    '''

    def __init__(self, inC : int, outC : int, hidC : int, WH : int, weights : int, bias : int):
        self.inC = inC
        self.outC = outC
        self.hidC = hidC
        self.WH = WH

        self.weights = weights
        self.bias = bias

    def __call__(self, image : torch.Tensor, gt : torch.Tensor, learning_rate : float, loss_scale : float = 1) -> Union[float, torch.Tensor]:

        B, C, H, W = image.shape

        image = image.view(-1,)

        loss = torch.zeros((1, ))
        pred = torch.zeros((B, self.outC)).view(-1, )


        self.weights, self.bias, loss, pred =  ConvCUDA.CNN(
            B,
            image,
            self.weights,
            self.bias,
            gt, 
            loss,
            pred, 
            self.inC, 
            self.hidC, self.outC, H, self.WH, learning_rate, loss_scale)

        return loss, pred
