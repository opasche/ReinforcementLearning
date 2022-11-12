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
import torchvision.transforms as T

import math

import copy



class EnvWrapper(object):
    
    def __init__(self, env):
        super(EnvWrapper, self).__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
    
    
    def reset(self):
        return self.env.reset()
    
    
    def step(self, action):
        return self.env.step(action)
    
    
    def render(self, state):
        self.env.render()
    
    
    def close(self, state):
        self.env.close()
    
