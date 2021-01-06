from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.environments.ImageTSCSEnv import ImageTSCSEnv
from tscsRL.agents import ddpg
import numpy as np


env = ImageTSCSEnv(
    nCyl=4,
    kMax=0.45,
    kMin=0.35,
    nFreq=11,
    stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 5
params['decay_timesteps'] = 8000
params['num_episodes'] = 5
params['noise_scale'] = 1.2
params['save'] = True
params['plot_hpc'] = True
params['use_wandb'] = False
params['reward'] = []
params['lowest'] = []


name = 'ddpgImgReward4cyl'

agent = ddpg.ImageDDPGAgent(
    env.observation_space,
    env.action_space,
    params,
    name)

agent.learn(env)

