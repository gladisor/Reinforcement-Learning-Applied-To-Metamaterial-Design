from tscsRL.agents import ddpg, ddqn
from tscsRL.environments.RadiiTSCSEnv import ContinuousRadiiTSCSEnv, DiscreteRadiiTSCSEnv

## Change these depending on specified setting
ring_radii = [5.0, 7.1]
nCyl_ring =[11, 12]
core_radius = 3.2
name = 'run1'

env = DiscreteRadiiTSCSEnv(
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	ring_radii=ring_radii,
	nCyl_ring=nCyl_ring,
	core_radius=core_radius)

## Change these depending on specified setting
env.ep_len = 250
env.grid_size = 12.0

params = ddqn.default_params()
params['save_every'] = 100
params['decay_timesteps'] = 2000
params['num_episodes'] = 4
params['reward'] = []
params['lowest'] = []
params['save'] = True
params['plot_hpc'] = True
params['use_wandb'] = False

params['batch_size'] = 256

agent = ddqn.DDQNAgent(
	env.observation_space,
	env.action_space,
	params,
	name)

agent.learn(env)
