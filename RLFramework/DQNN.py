#!/usr/bin/env python
# coding: utf-8
"""
Part of the RLFramework (2022)

@author: Olivier C. Pasche
"""

import numpy as np
import gym
import random
import time
from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
from torch import Tensor 
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim

import math

import copy

# Generated data

# Model 1
class DQN_FC(nn.Module):
    """
    Customizable fully connected feedforward pytorch neural network.
    Input parameters are the input size, number of actions (output size) and a list of hidden layer sizes (nb of neurons).
    """
    def __init__(self, input_size:int, nb_actions:int, Hidden_vect:list[int]=[32,32],
                 activation=nn.ELU(alpha=1.0), p_drop=0):
        super(DQN_FC, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(p=p_drop)
        
        dims_in = [input_size] + Hidden_vect
        
        self.layers = nn.ModuleList([nn.Linear(dims_in[i], dout) for i, dout in enumerate(Hidden_vect)])
        self.lin_out = nn.Linear(dims_in[-1], nb_actions)
    
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.lin_out(x)
        return x


# Model 2
class DQN_Conv(nn.Module):
    """
    Network with two convolutions & poolings, then one fully conected hidden layer.
    Input parameters are the number of channels in the two convolutions and the fully connected layer size (nb of neurons).
    """
    def __init__(self, input_channels, nb_actions, nb_channels_one, nb_channels_two, nb_hidden):
        super(DQN_Conv, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, nb_channels_one, kernel_size=(3,3))
        self.conv2 = nn.Conv2d(nb_channels_one, nb_channels_two, kernel_size=(3,3))
        self.fc1 = nn.Linear(1280, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, nb_actions)

    def forward(self, x):
        x = F.elu(F.max_pool2d(self.conv1(x), kernel_size=2),1.)
        x = F.elu(F.max_pool2d(self.conv2(x), kernel_size=4),1.)
        x = F.sigmoid(self.fc1(x.view(-1,x.shape[1]*x.shape[2]*x.shape[3])))
        x = self.fc2(x) #F.sigmoid(self.fc2(x))
        return x

