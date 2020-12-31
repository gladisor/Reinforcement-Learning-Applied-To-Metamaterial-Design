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
params['batch_size'] = 256
params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 9000
params['use_wandb'] = True

name = 'ddqnGradientReward4cyl'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)