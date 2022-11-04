#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random
import time
from IPython.display import clear_output




class Q_agent(object):
    """Basic Q-learning with epsilon-greedy policy."""
    
    def __init__(self, n_states, n_actions,
                 greedy_eps = 1, 
                 exploration_decay_rate = 0.001, 
                 discount_rate = 0.99,
                 lr = 0.1,
                 min_exploration_rate = 0.001,
                 max_exploration_rate = 1):
        
        
        self.greedy_eps0 = greedy_eps
        self.greedy_eps = self.greedy_eps0
        self.n_states = n_states
        self.n_actions = n_actions
        
        self.exploration_decay_rate = exploration_decay_rate
        self.discount_rate = discount_rate
        self.lr = lr
        self.min_exploration_rate = min_exploration_rate
        self.max_exploration_rate = max_exploration_rate
        
        # initiliaze all Q-table values to 0
        self.Q = np.zeros((self.n_states, self.n_actions))
    
    
    def reset_Q(self, n_states, n_actions):
        self.Q = np.zeros(self.n_states, self.n_actions)
    
    
    def make_action(self, observation, exploit_only=False):
        r = random.uniform(0, 1)
        
        if r > self.greedy_eps or (exploit_only):
            # exploit 
            action = np.argmax(self.Q[observation, :])
            
        else:
            # explore (take a random action)
            action = np.random.randint(0,self.n_actions)
        
        return action
    
    
    def update_table(self, old_state, new_state, reward, action, episode, t=0):
        self.Q[old_state,action] = (1-self.lr)*self.Q[old_state,action] + self.lr*(reward + self.discount_rate * np.max(self.Q[new_state, :]) )
        
        # update greedy eps
        self.greedy_eps = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate * episode)
    
