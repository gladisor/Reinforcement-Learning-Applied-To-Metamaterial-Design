from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg
from tscsRL import utils

## Name of the run we want to evaluate
name = 'ddpg3cylGradient'

params = ddpg.default_params()
params['decay_timesteps'] = 1000

env = ContinuousGradientTSCSEnv(
	nCyl=3,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)