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



class EnvWrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(EnvWrapper, self).__init__(env)
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range
    
    
    def step(self, action):
        return self.env.step(action)
    
    
    def reset(self):
        return self.env.reset()
    
    
    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)
    
    
    def close(self):
        self.env.close()
    


class DoneRewardWrapper(EnvWrapper):
    
    def __init__(self, env, done_reward):
        super(DoneRewardWrapper, self).__init__(env)
        self.done_reward = done_reward
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            reward = self.done_reward
        return observation, reward, terminated, truncated, info










