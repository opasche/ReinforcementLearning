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

from DQNN import *
from ReplayMemory import *
from Preprocessor import *
from DQNAgent import *

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt



#For reproductibility of the report's results
torch.manual_seed(1)

def train(environement='CartPole-v0' ,n_episodes=10000, n_timesteps=int(1e6), 
                exploration_decay_rate = 0.001,
                discount_rate = 0.999,
                lr = 1e-3,
                min_exploration_rate = 0.01,#0.001,
                max_exploration_rate = 1):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment frozen lake
    env = gym.make(environement).unwrapped
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    
    #Declare objects
    NN = DQN_Conv(3, n_actions, 16, 32, 64)#.to(device)
    preprocessor = Preprocessor()
    replay_memory = ReplayMemory(100000,256)
    
    
    #create agent
    agent = DQNAgent(n_actions, exploration_decay_rate, 
                     discount_rate, lr,
                     min_exploration_rate, max_exploration_rate,
                     NN, replay_memory, target_update_lag=10)
    reward=None
    
    
    # create total reward
    reward_list = []
    
    #training of agent
    start_time = time.time()
    for episode in range(n_episodes):
        state = env.reset()
        new_image = env.render('rgb_array')#, close=True)
        old_image = new_image
        state = preprocessor.get_state(new_image, new_image, done = False, initial_screen=True)
        total_reward = 0
        for timestep in range(n_timesteps):
            
            #env.render()
            #print(observation)
            
            action = agent.eps_greedy_action(state)
            #print(agent.greedy_eps)
            _, reward, done, info = env.step(action)
            new_image = env.render('rgb_array')#, close=True)
            new_state = preprocessor.get_state(old_image, new_image, done, initial_screen=False)
            agent.store_experience(state, action, reward, new_state, done)
            agent.update_policy(episode)#, timestep)
            state = new_state
            old_image = new_image
            
            # sum up the number of rewards after n episodes
            total_reward += reward
            
            if done:
                #print("Episode finished after {} timesteps".format(timestep + 1))
                break
        
        agent.update_target(episode)
        reward_list.append(total_reward)
        if ((episode+1)%100==0):
            print(f"---- Episodes {episode-98} to {episode+1} finished in {time.time() - start_time} seconds ----")
            start_time = time.time()
        if ((episode+1)%1000==0):
            agent.save_weights("convDQN_cartpole_weights_traintemp.pt")
    
    env.close()
    
    #print(np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.figure(figsize=(14,10))
    #plt.plot((np.arange(n_episodes) + 1), np.cumsum(reward_list)/(np.arange(n_episodes) + 1))
    plt.plot((np.arange(n_episodes) + 1), reward_list)
    plt.plot(np.convolve(np.array(reward_list), np.ones((20,))/20, mode='valid'))
    plt.show()
    return agent, reward_list



def play(agent, environement='CartPole-v0', n_episodes=10, n_timesteps=int(1e6), plot_rewards=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #create environment frozen lake
    env = gym.make(environement).unwrapped
    
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    reward=None
    
    
    # create total reward
    reward_list = []
    
    #training of agent
    start_time = time.time()
    for episode in range(n_episodes):
        state = env.reset()
        new_image = env.render('rgb_array')
        old_image = new_image
        state = preprocessor.get_state(new_image, new_image, done = False, initial_screen=True)
        total_reward = 0
        for timestep in range(n_timesteps):
            
            env.render()
            #print(observation)
            
            action = agent.make_action(state)
            _, reward, done, info = env.step(action)
            new_image = env.render('rgb_array')
            new_state = preprocessor.get_state(old_image, new_image, done, initial_screen=False)
            state = new_state
            old_image = new_image
            
            # sum up the number of rewards after n episodes
            total_reward += reward
           
            if done:
                #print("Episode finished after {} timesteps".format(timestep + 1))
                break
        
        reward_list.append(total_reward)
        if ((episode+1)%100==0):
            print(f"---- Episodes {episode-98} to {episode+1} finished in {time.time() - start_time} seconds ----")
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



agent, reward_list = train(n_episodes=2000)
agent.save_weights("convDQN_cartpole_weights_last.pt")
replay_memory = agent.replay_memory
rew_play = play(agent, n_episodes=5)
print(rew_play)