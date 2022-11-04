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

from DQNN import *
from ReplayMemory import *
from Preprocessor import *



class DQNAgent(object):
    """Deep Q-learning agent with policy and target networks, and epsilon-greedy policy."""
    
    def __init__(self, n_actions,
                 exploration_decay_rate = 0.001,
                 discount_rate = 0.999, lr = 1e-3,
                 min_exploration_rate = 0.01, max_exploration_rate = 1,
                 NN=None, preprocessor=Preprocessor(), replay_memory=ReplayMemory(100000,256), target_update_lag=10):
        
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.n_actions = n_actions
        
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.greedy_eps = self.max_exploration_rate
        
        self.discount_rate = discount_rate
        self.lr = lr
        
        self.PolicyNN = NN#.to(self.device)
        self.TargetNN = copy.deepcopy(self.PolicyNN)#.to(self.device)
        self.TargetNN.eval()
        #self.criterion = nn.L1Loss()#nn.CrossEntropyLoss() nn.L1Loss() nn.MSELoss()
        #self.criterion = lambda fx, y: (fx-y).abs().mean()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.PolicyNN.parameters(), lr=self.lr)
        
        self.preprocessor = preprocessor
        self.replay_memory = replay_memory
        self.target_update_lag = target_update_lag
        
        
    
    
    
    def make_action(self, state):
        
        # exploit
        processed = self.preprocessor.process(state)
        with torch.no_grad():
            pred = self.PolicyNN(processed)[0]
            action = pred.argmax().item()
            
        return action
    
    
    def eps_greedy_action(self, state):
        r = random.uniform(0, 1)
        
        if r > self.greedy_eps:
            # exploit
            processed = self.preprocessor.process(state)
            with torch.no_grad():
                pred = self.PolicyNN(processed)[0]
                action = pred.argmax().item()
            
        else:
            # explore (take a random action)
            action = np.random.randint(0,self.n_actions)
        
        return action
    
    
    def store_experience(self, state, action, reward, new_state, done):
        self.replay_memory.add_experience(state, action, reward, new_state, done, self.preprocessor)
    
    
    def update_policy(self, episode):#, t=0
        #self.Q[old_state,action] = (1-self.lr)*self.Q[old_state,action] + self.lr*(reward + self.discount_rate * np.max(self.Q[new_state, :]) )
        X, y, actions = self.replay_memory.make_batch(self.TargetNN, self.discount_rate)
        
        output = self.PolicyNN(X)#model(train_input.narrow(0, b, mini_batch_size))
        #self.PolicyNN(X).gather(dim=1, index=actions.unsqueeze(-1))
        chosen_output = torch.gather(output, 1, actions.view(-1,1)) #,torch.tensor([[0],[1]]))
        loss = self.criterion(chosen_output.view(-1,1), y.view(-1,1))#train_target.narrow(0, b, mini_batch_size))
        self.optimizer.zero_grad() #optimizer.zero_grad() ???? cours
        loss.backward()
        self.optimizer.step()
        
        # update greedy eps
        self.greedy_eps = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
    
    
    def update_target(self, epoch=None):
        
        if epoch:
            if((epoch+1)%self.target_update_lag == 0):
                self.TargetNN.load_state_dict(self.PolicyNN.state_dict())
                self.TargetNN.eval()
        else:
            self.TargetNN.load_state_dict(self.PolicyNN.state_dict())
            self.TargetNN.eval()
    
    
    def save_weights(self, filename="DQN_cartpole_weights.pt", folder="model_weights/"):
        torch.save(self.PolicyNN.state_dict(), folder+filename)
    
    
    def load_weights(self, filename="DQN_cartpole_weights.pt", folder="model_weights/"):
        self.PolicyNN.load_state_dict(torch.load(folder+filename))
        self.TargetNN.load_state_dict(torch.load(folder+filename))
        self.TargetNN.eval()
    
