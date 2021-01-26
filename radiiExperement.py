from tscsRL.agents import ddpg, ddqn
from tscsRL.environments.RadiiTSCSEnv import ContinuousRadiiTSCSEnv, DiscreteRadiiTSCSEnv
import torch

env = DiscreteRadiiTSCSEnv(
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	ring_radii=[5.0, 7.1],
	nCyl_ring=[11, 12],
	core_radius=3.2)

env.ep_len = 250
env.grid_size = 12

env.radii = torch.ones(1, env.design_M) * env.max_radii
env.renderIMG(env.radii)

# params = ddqn.default_params()
# params['save_every'] = 100
# params['decay_timesteps'] = 2000
# params['num_episodes'] = 2500
# params['use_wandb'] = True

# params['batch_size'] = 256

# name = 'ddqn_3.2core_3ring'

# agent = ddqn.DDQNAgent(
# 	env.observation_space,
# 	env.action_space,
# 	params,
# 	name)

# agent.learn(env)

# params = ddpg.default_params()
# params['save_every'] = 50
# params['decay_timesteps'] = 2000
# params['num_episodes'] = 2500
# params['use_wandb'] = True

# params['noise_scale'] = 0.25
# params['noise_scale_end'] = 0.001
# params['batch_size'] = 64

# name = 'ddpgRadii_19cyl_2000decay'

# agent = ddpg.DDPGAgent(
#     env.observation_space,
#     env.action_space,
#     params,
#     name)

