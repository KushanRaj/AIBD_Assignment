import numpy as np
import math
import torch
from CUDAConv import MyNetwork
from typing import Union

#For Inititalisation of weights
def _calculate_fan_in_and_fan_out(array : torch.Tensor) -> Union[int, int]:
    dimensions = len(array.shape)

    num_input_fmaps = array.shape[1]
    num_output_fmaps = array.shape[0]
    receptive_field_size = 1
    if dimensions > 2:
        for s in array.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def _calculate_correct_fan(array : torch.Tensor, mode : str) -> int:

    mode = mode.lower()
    
    fan_in, fan_out = _calculate_fan_in_and_fan_out(array)
    return fan_in if mode == 'fan_in' else fan_out

def kaiming_uniform_(array : torch.Tensor, mode : str = 'fan_in') -> torch.Tensor:
   
    fan = _calculate_correct_fan(array, mode)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  
    
    return array.uniform_(-bound, bound)

class Conv2D:
    """
    Conv2D (CPU & CUDA Version) 
    -----------------------------
    !! to match CUDA code we take kernel size as 2 and stride 2 with padding 0

    Parameters
    ---------------
    inC : int - Number of features of input
    outC : int - Number of features of outputs
    bias : bool - Whether the convolution has a bias parameter

    kernel : int = 2
    stride : int = 2
    padding : int = 0

    """

    def __init__(self, inC : int , outC : int, bias : bool = True) -> None:
        
        k = 2
        s = 2
        p = 0

        self.inC = inC
        self.outC = outC
        self.K = k
        self.S = s

        weight = torch.zeros((outC,inC,k,k))

        self.weight = kaiming_uniform_(weight)
        if bias:
            bias = torch.zeros((outC, ))
            fan_in, _ = _calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                bias.uniform_(-bound, bound)

            self.bias = bias

    def load_params(self, params : dict) -> None:
        """
        Loads params given a parameter dictionary
        
        params - dict : Dictionary containing Parameters
        """


        self.weight = params['weight']
        self.bias = params['bias']
        
    def params(self) -> dict:
        """
        Returns dictionary of Parameters
        """

        return {'weight' : self.weight, 'bias' : self.bias}

    def __call__(self, X : torch.Tensor) ->  torch.Tensor:

        """Computes convolution on input
        
        Parameters
        -------------
        X - torch.Tensor : input
        """
        

        B, C, H, W = X.shape

        out = torch.zeros((B, self.outC, H//2, W//2))

        for b in range(B):
            for oc in range(self.outC):
                for c in range(C):
                    for p in range((H//2) * (W//2)):
                        for k in range(self.K*self.K):
                            og_p = (p % (W//2)) * 2 + (2 * H * (p//(W//2)))
                            out[b, oc, p//(H//2), p % (W//2)] += X[b, c, (og_p//H) + (k//self.K), (og_p % H) + (k % self.K)] * self.weight[oc, c, k//self.K, k % self.K]
                            if (c == 0 and k == 0):
                                out[b, oc, p//(H//2), p % (W//2)] += self.bias[oc]

        return out

    def backward(self, X : torch.Tensor, out : torch.Tensor, learning_rate : float):

        """Computes grad for the weight and previous layer given graph
        
        Parameters
        -------------
        X - torch.Tensor : input
        out - torch.Tensor : backprop from next layer
        learning_rate - float : learning rate

        """

        B, C, H, W = X.shape

        grad = torch.zeros_like(self.weight)
        grad_bias = torch.zeros_like(self.bias)

        backprop = torch.zeros_like(X)

        for oc in range(self.outC):
            for c in range(C):
                for b in range(B):
                    for p in range((H//2) * (W//2)):
                        for ky in range(self.K):
                            for kx in range(self.K):
                                og_p = (p % (W//2)) * 2 + (2 * H * (p//(W//2)))
                                grad[oc, c, ky, kx] += X[b, c, (og_p//H) + ky, (og_p % H) + kx] * out[b, oc, p//(H//2), p % (W//2)]
                                backprop[b, c, (og_p//H) + ky, (og_p % H) + kx] += out[b, oc, p//(H//2), p % (W//2)] * self.weight[oc, c, ky, kx]
                                if (kx == 0 and ky == 0 and c == 0):
                                    grad_bias[oc] += out[b, oc, p//(H//2), p % (W//2)]

        self.weight = self.weight - learning_rate  * grad
        self.bias = self.bias - learning_rate * grad_bias


        return backprop

class ReLU:
    """
    Rectified Linear Unit Activation Function
    --------------------------------------------
    Acts as gate that only forwards positive values
    """
    def __init__(self,):
        pass

    def __call__(self, X : torch.Tensor) -> torch.Tensor:
        """
        Performs relu on input

        Parameters
        -------------
        X - torch.Tensor : input
        """


        return (X > 0 ) * X

    def backward(self, X: torch.Tensor, out : torch.Tensor) -> torch.Tensor:
        """Computes the backpropagation for previous layer given the graph
        
        Parameters
        -------------
        X - torch.Tensor : input
        out - torch.Tensor : backprop from next layer
        
        """

        return (X > 0) * out

class Pool2D:
    """
    MaxPool Function
    --------------------------------------------
    Pools information in sliding windows and forwards only the maximum value

    !! to match CUDA code we take kernel size as 2 and stride 2 with dilation 1
    """
    def __init__(self,):
        self.inds = None

    def __call__(self, X : torch.Tensor) -> torch.Tensor:

        """Computes MaxPool on input

        Parameters
        -------------
        X - torch.Tensor : input
        """

        B, C, H, W = X.shape

        out = torch.zeros((B, C, H//2, W//2))
        inds = torch.zeros_like(X)

        for b in range(B):
            for c in range(C):
                for p in range((H//2) * (W//2)):
                    og_p = (p % (W//2)) * 2 + (2 * H * (p//(W//2)))
                    vals = X[b, c, og_p//H : (og_p//H) + 2, (og_p % H) : (og_p % H) + 2].view(4, )
                    where = np.argmax(vals)
                    out[b, c, p//(H//2), p % (W//2)] = vals[where]
                    inds[b, c, (og_p//H) + (where//2), (og_p % H) + (where % 2)] = 1
                    
        
        self.inds = inds

        return out

    def backward(self, X : torch.Tensor, out : torch.Tensor) -> torch.Tensor:

        """Computes the backpropagation for previous layer given the graph
        
        Parameters
        -------------
        X - torch.Tensor : input
        out - torch.Tensor : backprop from next layer
        
        """

        assert type(self.inds) == torch.Tensor, "Please call forward pass first"

        backprop = torch.zeros_like(X)


        backprop[torch.where(self.inds == 1)] = out.view(-1, )

        return backprop

class Loss:

    """
    Categorical Cross Entropy
    """

    def __init__(self,):
        pass

    def __call__(self, X : torch.Tensor, gt : torch.Tensor, loss_scale : float = 1) -> Union[float, torch.Tensor, torch.Tensor]:

        """Compute Categorical Cross Entropy on Logits
        Returns loss, and outputs along woth backpropagation for previous layer

        Parameters
        -------------

        X - torch.Tensor : input
        gt - torch.Tensor : ground truth
        
        """

        B, C, _, _ = X.shape
        pred = X.view(B, C)

        probs = torch.exp(pred)/torch.exp(pred).sum(-1, keepdims=True)

        loss = -torch.log(probs[torch.arange(B), gt]).mean()

        backprop = (1/B) * (probs) * loss_scale

        backprop[torch.arange(B), gt] = (1/B) * (probs[torch.arange(B), gt] - 1) * loss_scale

        return loss, backprop.view(X.shape), pred

class Model:

    """
    Convolutional Neural Network for Classification(CPU & CUDA Version) 
    -----------------------------
    !! to match CUDA code we take kernel size as 2 and stride 2 with padding 0

    Parameters
    ---------------
    device : str - Whether to use CUDA or CPU
    inC : int - Number of features of input
    outC : int - Number of features of output 
    hidC : int - Number of hidden features
    WH : int - kernel size

    """
    def __init__(self, device : str, inC : int, outC : int, hidC : int, WH : int = 2) -> None:

        self.device = device

        if device == 'cuda':
    
            conv1 = Conv2D(1, 32)
            weight1 = conv1.weight
            bias1 = conv1.bias

            conv2 = Conv2D(32, 10)
            weight2 = conv2.weight
            bias2 = conv2.bias

            conv3 = Conv2D(10, 10)
            weight3 = conv3.weight
            bias3 = conv3.bias

            weights = torch.cat([weight1.view(-1, ), weight2.view(-1, ), weight3.view(-1, ), ])
            bias = torch.cat([bias1.view(-1, ), bias2.view(-1, ), bias3.view(-1, ), ])

            self.net = MyNetwork(inC, outC, hidC, WH, weights, bias)

        else:

            self.conv1 = Conv2D(1, 32)
            self.conv2 = Conv2D(32, 10)
            self.conv3 = Conv2D(10, 10)
            self.pool1 = Pool2D()
            self.pool2 = Pool2D()
            self.pool3 = Pool2D()
            self.activation = ReLU()

            self.criterion = Loss()

    def parameters(self) -> int:

        """Returns Parameter Count
        """

        if self.device == 'cuda': 
            weights = self.net.__dict__['weights']
            bias = self.net.__dict__['bias']
            return weights.view(-1).shape[0] + bias.view(-1).shape[0] 
        else:
            keys = sorted(k for k in self.__dict__.keys() if 'conv' in k)
            params = 0

            for i in keys:
                m_params = self.__dict__[i].params()
                for weight in m_params.values():
                    params += weights.view(-1,).shape[0]


        return params

    def state_dict(self) -> dict:
        """Return dict containing parameters of the model
        """


        params = {}

        if self.device == 'cuda': 
            
            
            params['weights'] = self.net.__dict__['weights']
            params['bias'] = self.net.__dict__['bias']

            

        else:

            keys = sorted(k for k in self.__dict__.keys() if 'conv' in k)

            for i in keys:
                params[i] = self.__dict__[i].params()
        
        return params

    
    def load_state_dict(self, params : dict) -> None:
        """Load paramters according to given paramter dictionary
        
        Parameters
        -------------
        params : dict - Dictionary of paramters
        """



        if self.device == 'cuda': 
            
            params = {}
            self.net.__dict__['weights'] = params['weights'] 
            self.net.__dict__['bias'] = params['bias']

        else:

            keys = sorted(k for k in self.__dict__.keys() if 'conv' in k)

            for i in keys:
                self.__dict__[i].load_params(params[i])


    def __call__(self, X : torch.Tensor, gt : torch.Tensor, learning_rate : float = 0, loss_scale : float = 1) -> Union[float, torch.Tensor]:

        """Classify input with the CNN

        Parameters
        -------------
        X - torch.Tensor : input
        gt - torch.Tensor : ground torch
        learning_rate - float : learning rate
        
        """


        if self.device == 'cuda':

            loss, pred = self.net(X, gt, learning_rate, loss_scale)
            loss = loss.item()

        else:

            out1 = self.conv1(X)
            out2 = self.activation(out1)
            out3 = self.pool1(out2)
            out4 = self.conv2(out3)
            out5 = self.activation(out4)
            out6 = self.pool2(out5)
            output = self.conv3(out6)

            loss, backprop, pred = self.criterion(output, gt, loss_scale)

            if learning_rate > 0:

                backprop = self.conv3.backward(out6, backprop, learning_rate)
                backprop = self.pool2.backward(out5, backprop)
                backprop = self.activation.backward(out4, backprop)
                backprop = self.conv2.backward(out3, backprop, learning_rate)
                backprop = self.pool1.backward(out2, backprop)
                backprop = self.activation.backward(out1, backprop)
                self.conv1.backward(X, backprop, learning_rate)

        return loss, pred

if __name__ == "__main__":

    model = Model('cuda', 1, 10, 32, 2)
    print(model(torch.randn(8, 1, 32, 32), torch.randint(0, 10, (8, )).int(), 0))


