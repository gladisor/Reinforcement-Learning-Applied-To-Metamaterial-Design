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
	nCyl=4,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 9000
params['noise_scale'] = 1.2
params['action_range'] = env.stepSize
params['save_data'] = False
params['use_wandb'] = False

name = 'test_speedup'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)