import gym
from gym.spaces import Box, Discrete
import numpy as np
import matplotlib.pyplot as plt
import random, time
import ray
from ray.rllib.agents.dqn import DQNTrainer, DEFAULT_CONFIG
from ray import tune

config = DEFAULT_CONFIG.copy()
config['env'] = "CartPole-v0"
config['num_workers'] = tune.grid_search([2, 3, 4, 5])
config['num_gpus'] = 1
config['num_cpus_per_worker'] = 1
config['sample_async'] = True
config['framework'] = 'torch'
config['model']['fcnet_hiddens'] = [100, 100]

stop = {
    'timesteps_total':50_000,
    ## If mean reward is 10
    'episode_reward_mean':200.0
}

analysis = tune.run(
    DQNTrainer,
    config=config,
    stop=stop,
    verbose=1)