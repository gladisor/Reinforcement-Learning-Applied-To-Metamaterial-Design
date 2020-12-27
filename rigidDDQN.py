from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.agents import ddqn

env = DiscreteTSCSEnv(
	nCyl=4,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddqn.default_params()
params['n_hidden'] = 2
params['h_size'] = 128
params['batch_size'] = 256

params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 8500
params['eps_end'] = 0.05
params['target_update'] = 10
params['save_data'] = False
params['use_wandb'] = True

name = 'test_ddqn_4cyl'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)