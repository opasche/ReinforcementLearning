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
    Fully connected network with two hidden layers.
    Input parameters are the sizes of the layers (nb of neurons).
    Takes as input data two channels images from MNIST and aims at finding which channel contains the largest digit.
    """
    def __init__(self, input_size, nb_actions, nb_hidden_one, nb_hidden_two):
        super(DQN_FC, self).__init__()
        self.fc1 = nn.Linear(input_size, nb_hidden_one)
        self.fc2 = nn.Linear(nb_hidden_one, nb_hidden_two)
        self.fc3 = nn.Linear(nb_hidden_two, nb_actions)


    def forward(self, x):
        x = F.elu(self.fc1(x),1.)
        x = F.elu(self.fc2(x),1.)
        x = self.fc3(x) #F.sigmoid(self.fc3(x))

        return x


# Model 2
class DQN_Conv(nn.Module):
    """
    Network with two convolutions & poolings, then one fully conected hidden layer.
    Input parameters are the number of channels in the two convolutions and the fully connected layer size (nb of neurons).
    Takes as input data two channels images from MNIST and aims at finding which channel contains the largest digit.
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

