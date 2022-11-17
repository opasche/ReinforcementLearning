#!/usr/bin/env python
# coding: utf-8
"""
To use with the RLFramework (2022)

@author: Olivier C. Pasche
"""


import numpy as np
import gym
import random
import time
import sys
from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
from torch import Tensor 
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim

import math

sys.path.append('../../../')
from RLFramework.DQNN import *
from RLFramework.ReplayMemory import *
from RLFramework.Preprocessor import *
from RLFramework.Agents import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt




# For reproductibility
torch.manual_seed(1)

def train(environement='CartPole-v1', n_episodes=10000, n_timesteps=500, 
          Hidden_vect=[32,32], discount_rate = 0.999, lr = 1e-3,
          max_exploration_rate = 1, exploration_decay_rate = 0.001, min_exploration_rate = 0.01,#0.001,
          warm_start_path=None, render_mode="rgb_array_list", **kwarg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment
    env = gym.make(environement, render_mode=render_mode, **kwarg)#.env
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    
    #Declare objects
    NN = DQN_FC(state_shape[0], n_actions, Hidden_vect=Hidden_vect,
                activation=nn.ELU(alpha=1.0), p_drop=0)#.to(device)
    preprocessor = TensorPreprocessor()
    replay_memory = ReplayMemory(100000,256)
    
    
    #create agent
    agent = DQNAgent(n_actions, NN, discount_rate, lr,
                     max_exploration_rate, exploration_decay_rate, min_exploration_rate, 
                     preprocessor, replay_memory, target_update_lag=10)
    
    reward_list = agent.train(env, n_episodes=n_episodes, max_timesteps=n_timesteps,
                              checkpoint_path="./model_weights/checkpoints_last/",
                              warm_start_path=warm_start_path, verbatim=1, render_every=None)
    
    env.close()
    
    #print(np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.figure(figsize=(14,10))
    #plt.plot((np.arange(n_episodes) + 1), np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.plot((np.arange(n_episodes) + 1), reward_list)
    plt.plot(np.convolve(np.array(reward_list), np.ones((50,))/50, mode='valid'))
    plt.show()
    return agent, reward_list



def play(agent, environement='CartPole-v1', n_episodes=5, n_timesteps=1000, plot_rewards=False,
         render_mode="human", **kwarg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment
    env = gym.make(environement, render_mode=render_mode, **kwarg).env
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    reward_list = agent.play(env, n_episodes=n_episodes, max_timesteps=n_timesteps,
                             verbatim=1, render_every=1)
    
    env.close()
    
    if plot_rewards:
        #print(np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
        plt.figure(figsize=(14,10))
        #plt.plot((np.arange(n_episodes) + 1), np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
        plt.plot((np.arange(n_episodes) + 1), reward_list)
        plt.plot(np.convolve(np.array(reward_list), np.ones((20,))/20, mode='valid'))
        plt.show()
    return reward_list



agent, reward_list = train(environement='CartPole-v1', n_episodes=5000, n_timesteps=500, 
                           Hidden_vect=[32,24], discount_rate = 0.999, lr = 1e-3,
                           max_exploration_rate = 1, exploration_decay_rate = 0.001, min_exploration_rate = 0.01,#0.001,
                           warm_start_path=None, render_mode="rgb_array")
agent.save("model_weights/DQN_cartpole_weights_last.pt")
replay_memory = agent.replay_memory

#agent.load("model_weights/DQN_cartpole_weights_best_32_24.pt")

rew_play = play(agent, environement='CartPole-v1', n_episodes=5, n_timesteps=1000, plot_rewards=False, render_mode="human")
print(rew_play)




