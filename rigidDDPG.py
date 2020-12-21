from tscsRL.environments.ContinuousTSCSEnv import ContinuousTSCSEnv
from tscsRL.agents import ddpg

env = ContinuousTSCSEnv(
	nCyl=2,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 100
params['decay_timesteps'] = 100
params['num_episodes'] = 120
params['noise_scale'] = 1.1
params['action_range'] = env.stepSize
params['save_data'] = False
params['use_wandb'] = True

name = 'test_ddpg'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)