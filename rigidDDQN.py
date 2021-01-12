from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import DiscreteGradientTSCSEnv
from tscsRL.agents import ddqn

# env = DiscreteTSCSEnv(
# 	nCyl=4,
# 	kMax=0.45,
# 	kMin=0.35,
# 	nFreq=11,
# 	stepSize=0.5)

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
params['num_episodes'] = 8500
params['reward'] = []
params['lowest'] = []
params['save'] = True
params['plot_hpc'] = True
params['use_wandb'] = False

name = 'ddqnNoGradient4cyl_run1'
# name = 'ddqnGradient4cy_run1'
# name = 'ddqnGradientReward4cyl_run1'

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)