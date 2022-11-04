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

from DQNN import *
from ReplayMemory import *
from Preprocessor import *
from DQNAgent import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt




#For reproductibility of the report's results
torch.manual_seed(1)

def train(environement='CartPole-v1', n_episodes=10000, n_timesteps=500, 
                exploration_decay_rate = 0.001,
                discount_rate = 0.999,
                lr = 1e-3,
                min_exploration_rate = 0.01,#0.001,
                max_exploration_rate = 1,
                **kwarg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment
    env = gym.make(environement, **kwarg)#.env
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    
    #Declare objects
    #NN = DQN_FC(state_shape[0], n_actions, 64, 32)#.to(device)
    NN = DQN_FC(state_shape[0], n_actions, 32, 24)#.to(device)
    preprocessor = Preprocessor()
    replay_memory = ReplayMemory(100000,256)
    
    
    #create agent
    agent = DQNAgent(n_actions, exploration_decay_rate, 
                     discount_rate, lr,
                     min_exploration_rate, max_exploration_rate,
                     NN, preprocessor, replay_memory, target_update_lag=10)
    
    agent.load_weights("DQN_cartpole_weights_last.pt")
    
    reward=None
    
    
    # create total reward
    reward_list = []
    
    #training of agent
    start_time = time.time()
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        for timestep in range(n_timesteps):
            
            #env.render()
            #print(observation)
           
            if done:
                #print("Episode finished after {} timesteps".format(timestep + 1))
                break
            
            action = agent.eps_greedy_action(state)
            #print(agent.greedy_eps)
            new_state, reward, done, truncated, info = env.step(action)
            agent.store_experience(state, action, reward, new_state, done)
            agent.update_policy(episode)#, timestep)
            state = new_state
            
            # sum up the number of rewards after n episodes
            total_reward += reward
        
        agent.update_target(episode)
        reward_list.append(total_reward)
        if ((episode+1)%100==0):
            print(f"---- Episodes {episode-98} to {episode+1} finished in {(time.time() - start_time):.2f} seconds ----")
            start_time = time.time()
        if ((episode+1)%1000==0):
            agent.save_weights("DQN_cartpole_weights_traintemp_"+str(episode+1)+".pt")
    
    env.close()
    
    #print(np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.figure(figsize=(14,10))
    #plt.plot((np.arange(n_episodes) + 1), np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.plot((np.arange(n_episodes) + 1), reward_list)
    plt.plot(np.convolve(np.array(reward_list), np.ones((50,))/50, mode='valid'))
    plt.show()
    return agent, reward_list



def play(agent, environement='CartPole-v1', n_episodes=5, n_timesteps=1000, plot_rewards=False,
                **kwarg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment
    env = gym.make(environement, **kwarg).env
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    reward=None
    
    
    # create total reward
    reward_list = []
    
    #training of agent
    start_time = time.time()
    for episode in range(n_episodes):
        state, info = env.reset()
        total_reward = 0
        done = False
        for timestep in range(n_timesteps):
            
            env.render()
            #print(observation)
           
            if done:
                #print("Episode finished after {} timesteps".format(timestep + 1))
                break
            
            action = agent.make_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            state = new_state
            
            # sum up the number of rewards after n episodes
            total_reward += reward
        
        reward_list.append(total_reward)
        if ((episode+1)%100==0):
            print(f"---- Episodes {episode-98} to {episode+1} finished in {(time.time() - start_time):.2f} seconds ----")
            start_time = time.time()
    
    env.close()
    
    if plot_rewards:
        #print(np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
        plt.figure(figsize=(14,10))
        #plt.plot((np.arange(n_episodes) + 1), np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
        plt.plot((np.arange(n_episodes) + 1), reward_list)
        plt.plot(np.convolve(np.array(reward_list), np.ones((20,))/20, mode='valid'))
        plt.show()
    return reward_list



agent, reward_list = train(n_episodes=10000)
agent.save_weights("DQN_cartpole_weights_last.pt")
replay_memory = agent.replay_memory
rew_play = play(agent, n_episodes=5)
print(rew_play)
