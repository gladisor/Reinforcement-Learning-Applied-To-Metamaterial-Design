from tscsRL.environments.DiscreteTSCSEnv import DiscreteTSCSEnv
from tscsRL.agents import ddqn

env = DiscreteTSCSEnv(
	nCyl=2,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddqn.default_params()
params['save_every'] = 100
params['decay_timesteps'] = 100
params['eps_end'] = 0.05
params['target_update'] = 10
params['num_episodes'] = 120
params['save_data'] = False
params['use_wandb'] = True

name = 'test_ddqn_fixed'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)