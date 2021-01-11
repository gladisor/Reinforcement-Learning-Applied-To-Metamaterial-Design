from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg

# env = ContinuousTSCSEnv(
# 	nCyl=2,
# 	kMax=0.45,
# 	kMin=0.35,
# 	nFreq=11,
# 	stepSize=0.5)

env = ContinuousGradientTSCSEnv(
	nCyl=6,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5
	)

env.grid_size = 8.0

params = ddpg.default_params()
params['save_every'] = 1000
params['decay_timesteps'] = 8000
params['num_episodes'] = 9000
params['noise_scale'] = 1.0
params['action_range'] = env.stepSize
params['save_data'] = False
params['use_wandb'] = True

# params['actor_n_hidden'] = 3


test = 'M6_8grid_NS1'
name = 'test_' + test

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)