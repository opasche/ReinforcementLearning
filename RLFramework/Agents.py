#!/usr/bin/env python
# coding: utf-8
"""
Part of the RLFramework (2022)

@author: Olivier C. Pasche
"""


import numpy as np
import random
import time
from IPython.display import clear_output

# For Deep Q-Agents
import torch
from torch import Tensor 
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from torch import optim

import math

import copy

from RLFramework.DQNN import *
from RLFramework.ReplayMemory import *
from RLFramework.Preprocessor import *



class RLAgent(object):
    """Generic RL Agent (with default epsilon-greedy exploratory policy)."""
    
    def __init__(self, n_actions,
                 discount_rate = 0.99,
                 lr = 0.1,
                 max_exploration_rate = 1,
                 exploration_decay_rate = 0.001, 
                 min_exploration_rate = 0.001,
                 preprocessor=Preprocessor()):
        
        super(RLAgent, self).__init__()
        
        self.n_actions = n_actions
        
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.greedy_eps = self.max_exploration_rate
        
        self.discount_rate = discount_rate
        self.lr = lr
        
        self.preprocessor = preprocessor
    
    
    def explore(self, state=None):
        # explore (take a random action)
        return np.random.randint(0, self.n_actions)
    
    
    def explore_subset(self, actions):
        # explore (take a random action among actions)
        action = np.random.choice(actions)
        if action not in np.arange(0, self.n_actions):
            raise ValueError
        return action
    
    
    def exploit(self, state):
        raise NotImplementedError
    
    
    def make_action(self, observation, exploit_only=False):
        r = random.uniform(0, 1)
        state = self.preprocessor.process(observation)
        
        if (r > self.greedy_eps) or (exploit_only):
            # exploit 
            action = self.exploit(state)
            
        else:
            # explore (take a random action)
            action = self.explore(state)
        
        return action
    
    
    def update_policy(self, old_state, new_state, reward, action, episode, done=False, t=0):
        raise NotImplementedError
    
    
    def update_greedy_eps(self, episode):
        # update greedy eps
        self.greedy_eps = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
    
    
    def save(self, filename, folder="./"):
        raise NotImplementedError
    
    
    def load(self, filename, folder="./"):
        raise NotImplementedError
    
    
    def train(self, env, n_episodes=10000, max_timesteps=500,
              checkpoint_path="./", warm_start_weights=None, verbatim=1, render_every=None):
        # Training proceidure with a gym-like environment.
        if warm_start_weights:
            self.load(warm_start_weights)
        
        # create total reward
        reward_list = []
        
        #training proceidure
        start_time = time.time()
        for episode in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            for timestep in range(max_timesteps):
                
                if render_every:
                    if episode%render_every==0:
                        env.render()
                        #print(observation)
               
                if done:
                    #print("Episode finished after {} timesteps".format(timestep + 1))
                    break
                
                action = self.make_action(state) #self.eps_greedy_action(state)
                #print(self.greedy_eps)
                new_state, reward, done, truncated, info = env.step(action)
                #self.store_experience(state, action, reward, new_state, done)
                self.update_policy(state, new_state, reward, action, episode, done=False, t=timestep)
                state = new_state
                
                # sum up the number of rewards after n episodes
                total_reward += reward
            
            #self.update_target_network(episode)
            reward_list.append(total_reward)
            if ((episode+1)%100==0) and (verbatim>0):
                print(f"---- Episodes {episode-98} to {episode+1} finished in {(time.time() - start_time):.2f} seconds ----")
                start_time = time.time()
            if checkpoint_path:
                if ((episode+1)%1000==0):
                    self.save(checkpoint_path+"DQN_cartpole_weights_traintemp_"+str(episode+1)+".pt")
        
        return(reward_list)
    
    
    def play(self, env, n_episodes=5, max_timesteps=1000, verbatim=1, render_every=1):
        # create total reward
        reward_list = []
        
        #training proceidure
        start_time = time.time()
        for episode in range(n_episodes):
            state, info = env.reset()
            total_reward = 0
            done = False
            for timestep in range(max_timesteps):
                
                if render_every:
                    if episode%render_every==0:
                        env.render()
                        #print(observation)
               
                if done:
                    #print("Episode finished after {} timesteps".format(timestep + 1))
                    break
                
                action = self.make_action(state, exploit_only=True)
                new_state, reward, done, truncated, info = env.step(action)
                state = new_state
                
                # sum up the number of rewards after n episodes
                total_reward += reward
            
            reward_list.append(total_reward)
            if ((episode+1)%100==0) and (verbatim>0):
                print(f"---- Episodes {episode-98} to {episode+1} finished in {(time.time() - start_time):.2f} seconds ----")
                start_time = time.time()
        
        return(reward_list)
    



class Q_agent(RLAgent):
    """Basic (one-step, off-policy, greedy) Q-learning with epsilon-greedy exploratory policy."""
    
    def __init__(self, n_states, n_actions,
                 discount_rate = 0.99,
                 lr = 0.1,
                 max_exploration_rate = 1,
                 exploration_decay_rate = 0.001, 
                 min_exploration_rate = 0.001,
                 preprocessor=Preprocessor()):
        
        super(Q_agent, self).__init__(n_actions=n_actions,
                     discount_rate = discount_rate, lr = lr,
                     max_exploration_rate = max_exploration_rate,
                     exploration_decay_rate = exploration_decay_rate, 
                     min_exploration_rate = min_exploration_rate,
                     preprocessor=preprocessor)
        
        self.n_states = n_states
        
        # initiliaze all Q-table values to 0
        self.reset_Q()
    
    
    def reset_Q(self):
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    
    def exploit(self, state):
        return np.random.choice(np.argwhere(self.Q[state, :]==np.max(self.Q[state, :])).reshape(-1,))
    
    
    # def make_action(self, observation, exploit_only=False):
    #     return super(Q_agent, self).make_action(observation=observation, exploit_only=exploit_only)
    
    
    def update_policy(self, old_state, new_state, reward, action, episode, done=False, t=0):
        self.Q[old_state,action] = (1-self.lr)*self.Q[old_state,action] + self.lr*(reward + self.discount_rate * np.max(self.Q[new_state, :]) )
        
        # update greedy eps
        self.update_greedy_eps(episode)
    
    
    def save(self, filename="Q_agent_table.npy", folder="./"):
        np.save(folder+filename, self.Q)
    
    
    def load(self, filename, folder="./"):
        self.Q = np.load(folder+filename)
    


class nstep_Q_agent(RLAgent):
    """To Do. n-step off-policy Sarsa sub optimal. n-step Q(sigma) A7.6 unifies: per decision with control (7.13)+(7.2) and Tree-backup A7.5. Eligibility traces improves efficiency of Q(sigma)."""
    
    def __init__(self, n_states, n_actions, n_steps=1):
        
        super(nstep_Q_agent, self).__init__()
        raise NotImplementedError
        
        



class DQNAgent(RLAgent):
    """Deep Q-learning agent with policy and target networks, and epsilon-greedy policy."""
    
    def __init__(self, n_actions,
                 discount_rate = 0.999, lr = 1e-3,
                 max_exploration_rate = 1, exploration_decay_rate = 0.001, min_exploration_rate = 0.01, 
                 NN=None, preprocessor=TensorPreprocessor(), replay_memory=ReplayMemory(100000,256), target_update_lag=10):
        
        super(DQNAgent, self).__init__(n_actions=n_actions,
                                       discount_rate = discount_rate, lr = lr,
                                       max_exploration_rate = max_exploration_rate,
                                       exploration_decay_rate = exploration_decay_rate, 
                                       min_exploration_rate = min_exploration_rate,
                                       preprocessor=preprocessor)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.PolicyNN = NN#.to(self.device)
        self.TargetNN = copy.deepcopy(self.PolicyNN)#.to(self.device)
        self.PolicyNN.eval()
        self.TargetNN.eval()
        
        #self.criterion = nn.L1Loss()#nn.CrossEntropyLoss() nn.L1Loss() nn.MSELoss()
        #self.criterion = lambda fx, y: (fx-y).abs().mean()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.PolicyNN.parameters(), lr=self.lr)
        
        self.replay_memory = replay_memory
        self.target_update_lag = target_update_lag
        
        self.last_episode = 0 
    
    
    def exploit(self, state):
        # exploit
        self.PolicyNN.eval()
        with torch.no_grad():
            pred = self.PolicyNN(state)[0]
            action = pred.argmax().item()
            
        return action
    
    
    def store_experience(self, state, action, reward, new_state, done):
        self.replay_memory.add_experience(self.preprocessor.process(state), action, reward,
                                          self.preprocessor.process(new_state), done)
    
    
    def update_policy(self, old_state, new_state, reward, action, episode, done=False, t=0):#, t=0
        
        self.store_experience(old_state, action, reward, new_state, done)
        
        if (episode != self.last_episode):
            self.update_target_network(episode-1)
        
        self.update_policy_network()
        
        # update greedy eps
        self.update_greedy_eps(episode)
        
        self.last_episode = episode
    
    
    def update_policy_network(self):
        #self.Q[old_state,action] = (1-self.lr)*self.Q[old_state,action] + self.lr*(reward + self.discount_rate * np.max(self.Q[new_state, :]) )
        X, y, actions = self.replay_memory.make_batch(self.TargetNN, self.discount_rate)
        
        self.PolicyNN.train()
        output = self.PolicyNN(X)#model(train_input.narrow(0, b, mini_batch_size))
        #self.PolicyNN(X).gather(dim=1, index=actions.unsqueeze(-1))
        chosen_output = torch.gather(output, 1, actions.view(-1,1)) #,torch.tensor([[0],[1]]))
        loss = self.criterion(chosen_output.view(-1,1), y.view(-1,1))#train_target.narrow(0, b, mini_batch_size))
        self.optimizer.zero_grad() #optimizer.zero_grad() ???? cours
        loss.backward()
        self.optimizer.step()
        self.PolicyNN.eval()
    
    
    def update_target_network(self, epoch=None):
        
        if epoch:
            if((epoch+1)%self.target_update_lag == 0):
                self.TargetNN.load_state_dict(self.PolicyNN.state_dict())
                self.TargetNN.eval()
        else:
            self.TargetNN.load_state_dict(self.PolicyNN.state_dict())
            self.TargetNN.eval()
    
    
    def save(self, filename="DQN_weights.pt", folder="./"):
        torch.save(self.PolicyNN.state_dict(), folder+filename)
    
    
    def load(self, filename, folder="./"):
        self.PolicyNN.load_state_dict(torch.load(folder+filename))
        self.TargetNN.load_state_dict(torch.load(folder+filename))
        self.PolicyNN.eval()
        self.TargetNN.eval()
    
    
    def eps_greedy_action(self, state):
        # DEPRECATED
        r = random.uniform(0, 1)
        
        if r > self.greedy_eps:
            # exploit
            processed = self.preprocessor.process(state)
            with torch.no_grad():
                pred = self.PolicyNN(processed)[0]
                action = pred.argmax().item()
            
        else:
            # explore (take a random action)
            action = self.explore()
        
        return action
    
