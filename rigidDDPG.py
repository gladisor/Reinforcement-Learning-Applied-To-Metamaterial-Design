from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg
import numpy as np

# env = ContinuousTSCSEnv(
# 	nCyl=4,
# 	kMax=0.45,
# 	kMin=0.35,
# 	nFreq=11,
# 	stepSize=0.5)

env = ContinuousGradientTSCSEnv(
	nCyl=10,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 10
params['noise_scale'] = 1.2
params['reward'] = []
params['lowest'] = []
params['invalid'] = []
params['save'] = True
params['plot_hpc'] = True
params['use_wandb'] = False

name = 'ddpgNoGradient10cyl_run2'
#name = 'ddpgGradient4cyl_run1'
#name = 'ddpgGradientReward4cyl_run1'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)