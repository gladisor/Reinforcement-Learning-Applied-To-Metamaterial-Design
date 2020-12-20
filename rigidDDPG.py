from tscsRL.environments.ContinuousTSCSEnv import ContinuousTSCSEnv
from tscsRL.agents.ddpg import DDPGAgent, default_params
import wandb

env = ContinuousTSCSEnv(
	nCyl=2,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = default_params()
params['save_every'] = 100
params['decay_timesteps'] = 1000
params['num_episodes'] = 1500

name = '2cylRigidDDPG'

agent = DDPGAgent(
	env.observation_space, 
	env.action_space, 
	env.stepSize, 
	params, 
	name)

wandb.init(project='tscs', config=params)

agent.learn(env)