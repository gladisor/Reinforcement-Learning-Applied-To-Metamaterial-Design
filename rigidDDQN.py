from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import DiscreteGradientTSCSEnv
from tscsRL.agents import ddqn

env = DiscreteGradientTSCSEnv(
	nCyl=4,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddqn.default_params()
params['n_hidden'] = 1
params['h_size'] = 128
params['batch_size'] = 256
params['lr'] = 0.001

params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 8500
params['target_update'] = 10
params['use_wandb'] = True

name = 'ddqnGradient4cyl0.45-0.35-8000decay'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)