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


class ReplayMemory(object):
    
    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.st = None
        self.at = None
        self.rtp1 = None
        self.stp1 = None
        self.is_done = None
        
    
    
    def add_experience(self, state, action, reward, new_state, done, preprocessor=None):
        
        if preprocessor is not None:
            state = preprocessor.process(state)
            new_state = preprocessor.process(new_state)
        
        if self.st is not None:
            
            if self.st.shape[0]>=self.capacity:
                self.st = torch.cat((self.st[-self.capacity+1:], state), dim=0)
                self.at = torch.cat((self.at[-self.capacity+1:], torch.Tensor([action])), dim=0)
                self.rtp1 = torch.cat((self.rtp1[-self.capacity+1:], torch.Tensor([reward])), dim=0)
                self.stp1 = torch.cat((self.stp1[-self.capacity+1:], new_state), dim=0)
                self.is_done = torch.cat((self.is_done[-self.capacity+1:], torch.Tensor([done]).type(torch.bool)), dim=0)
            else:
                self.st = torch.cat((self.st, state), dim=0)
                self.at = torch.cat((self.at, torch.Tensor([action])), dim=0)
                self.rtp1 = torch.cat((self.rtp1, torch.Tensor([reward])), dim=0)
                self.stp1 = torch.cat((self.stp1, new_state), dim=0)
                self.is_done = torch.cat((self.is_done, torch.Tensor([done]).type(torch.bool)), dim=0)
            
            
            
        else:#Checker si float
            self.st = state#torch.Tensor(state)#.to(self.device)
            self.at = torch.Tensor([action])#.to(self.device)
            self.rtp1 = torch.Tensor([reward])#.to(self.device)
            self.stp1 = new_state#torch.Tensor(new_state)#.to(self.device)
            self.is_done = torch.Tensor([done]).type(torch.bool)#.to(self.device)
        
    
    
    def make_batch(self, TargetNN, discount_rate):
        
        if self.st.shape[0]<self.batch_size:
            inds = range(self.st.shape[0])
        else:
            inds = sorted(random.sample(range(self.st.shape[0]),self.batch_size))
        
        X = self.st[inds]
        
        y = self.rtp1[inds] + discount_rate*TargetNN(self.stp1[inds]).max(1).values
        y[self.is_done[inds]] = self.rtp1[inds][self.is_done[inds]]
        
        actions = self.at[inds].long()
            
        return X, y, actions
    
#    def get_next(target_net, next_states):                
#        final_state_locations = next_states.flatten(start_dim=1) \
#            .max(dim=1)[0].eq(0).type(torch.bool)
#        non_final_state_locations = (final_state_locations == False)
#        non_final_states = next_states[non_final_state_locations]
#        batch_size = next_states.shape[0]
#        values = torch.zeros(batch_size).to(QValues.device)
#        values[non_final_state_locations] = target_net(non_final_states).max(dim=1)[0].detach()
#        return values
