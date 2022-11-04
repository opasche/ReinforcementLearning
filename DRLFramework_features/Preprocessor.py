#!/usr/bin/env python
# coding: utf-8
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



class Preprocessor(object):
    
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    def process(self, state):
        return torch.Tensor(state).view(1,-1)#.to(self.device)
    