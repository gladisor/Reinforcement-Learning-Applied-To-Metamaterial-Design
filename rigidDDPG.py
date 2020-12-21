from tscsRL.environments.ContinuousTSCSEnv import ContinuousTSCSEnv
from tscsRL.agents import ddpg
import wandb

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
params['save_data'] = False

name = 'test_run'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	env.stepSize, 
	params, 
	name)

logger = wandb.init(project='tscs', config=params, name=name)

agent.learn(env, logger)