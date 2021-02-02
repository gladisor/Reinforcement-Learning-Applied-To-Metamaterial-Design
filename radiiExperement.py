from tscsRL.agents import ddpg, ddqn
from tscsRL.environments.RadiiTSCSEnv import ContinuousRadiiTSCSEnv, DiscreteRadiiTSCSEnv

## Change these depending on specified setting
ring_radii = [3.1, 5.2, 7.3]
nCyl_ring = [9, 10, 11]
core_radius = 1.6

env = DiscreteRadiiTSCSEnv(
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	ring_radii=ring_radii,
	nCyl_ring=nCyl_ring,
	core_radius=core_radius)

# ## Change these depending on specified setting
env.ep_len = 250
env.grid_size = 9.0

state = env.reset()
env.renderIMG(env.radii)

params = ddqn.default_params()
params['save_every'] = 100
params['decay_timesteps'] = 2000
params['num_episodes'] = 2500
params['use_wandb'] = True

params['batch_size'] = 256

name = '3ring_1.6core'

agent = ddqn.DDQNAgent(
	env.observation_space,
	env.action_space,
	params,
	name)

agent.learn(env)
