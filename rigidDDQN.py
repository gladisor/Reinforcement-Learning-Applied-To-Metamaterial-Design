from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import DiscreteGradientTSCSEnv
from tscsRL.agents import ddqn

env = DiscreteTSCSEnv(
	nCyl=4,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddqn.default_params()
params['batch_size'] = 256
params['save_every'] = 500

# # M3
# params['decay_timesteps'] = 2000
# params['num_episodes'] = 2500

# M4
params['decay_timesteps'] = 8000
params['num_episodes'] = 8501

params['use_wandb'] = True

# env.ep_len = 200
# env.grid_size = 10.0

# name = 'test_ddqn_M10_grid10_ep200'
# name = 'ddqn_M4_ni_POR_2'
# name = 'ddqn_M10_grid10_eplen200'
# name = 'ddqn_thinshell_Ni_M3'
name = 'ddqn_thinshell_Ni_M4'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)