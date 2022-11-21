from __future__ import print_function

import sys, gym, time
import pygame
from gym.utils.play import play, PlayPlot

fps = 30

mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 1}

def callback(obs_t, obs_tp, action, reward, terminated, truncated, info):
    return [reward,]
plotter = PlayPlot(callback, 30 * 5, ["reward"])

play(gym.make("CartPole-v1", render_mode="rgb_array"), fps=fps, zoom=1.0, keys_to_action=mapping, callback=plotter.callback, noop=0)


