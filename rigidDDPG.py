from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg

env = ContinuousGradientTSCSEnv(
	nCyl=3,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 500
params['decay_timesteps'] = 2000
params['num_episodes'] = 2501
params['noise_scale'] = 1.2
params['use_wandb'] = True

name = 'ddpg_M3_thinshell_ni_rerun'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)