from tscsRL.agents import ddpg, ddqn
from tscsRL.environments.RadiiTSCSEnv import ContinuousRadiiTSCSEnv, DiscreteRadiiTSCSEnv

env = ContinuousRadiiTSCSEnv(
    kMax=0.45,
    kMin=0.35,
    nFreq=11)

env.ep_len = 200

# params = ddqn.default_params()
# params['save_every'] = 100
# params['decay_timesteps'] = 2000
# params['num_episodes'] = 2500
# params['use_wandb'] = True

# params['batch_size'] = 256

# name = 'ddqnRigidRadii_19cyl'

# agent = ddqn.DDQNAgent(
# 	env.observation_space,
# 	env.action_space,
# 	params,
# 	name)

params = ddpg.default_params()
params['save_every'] = 50
params['decay_timesteps'] = 2000
params['num_episodes'] = 2500
params['use_wandb'] = True

params['noise_scale'] = 0.25
params['noise_scale_end'] = 0.001
params['batch_size'] = 64

name = 'ddpgRadii_19cyl_2000decay'

agent = ddpg.DDPGAgent(
    env.observation_space,
    env.action_space,
    params,
    name)

agent.learn(env)