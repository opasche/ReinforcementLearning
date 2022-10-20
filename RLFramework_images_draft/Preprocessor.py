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
import torchvision.transforms as T

import math

import copy



class Preprocessor(object):
    
    def __init__(self):
        super(Preprocessor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    def get_state(self, old_image, new_image, done=False, initial_screen=False):
        if initial_screen or done:
            current_screen = self.process(new_image)
            black_screen = torch.zeros_like(current_screen)
            return black_screen
        else:
            s1 = self.process(old_image)
            s2 = self.process(new_image)
            return s2 - s1
    
    
    def process(self, screen):
        screen = screen.transpose((2, 0, 1))#self.render('rgb_array').transpose((2, 0, 1)) # PyTorch expects CHW
        screen = self.crop_screen(screen)
        return self.transform_screen_data(screen)
    
    
    def crop_screen(self, screen):
        screen_height = screen.shape[1]
        
        # Strip off top and bottom
        top = int(screen_height * 0.4)
        bottom = int(screen_height * 0.8)
        screen = screen[:, top:bottom, :]
        return screen
    
    
    def transform_screen_data(self, screen):       
        # Convert to float, rescale, convert to tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        
        # Use torchvision package to compose image transforms
        resize = T.Compose([
            T.ToPILImage()
            ,T.Resize((40,90))
            ,T.ToTensor()
        ])
        
        return resize(screen).unsqueeze(0)#.to(self.device) # add a batch dimension (BCHW)
    
    
    
    