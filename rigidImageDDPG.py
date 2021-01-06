from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.environments.ImageTSCSEnv import ImageTSCSEnv
from tscsRL.agents import ddpg

env = ImageTSCSEnv(
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
params['use_wandb'] = True

name = 'ddpgGradientReward4cyl'

agent = ddpg.ImageDDPGAgent(
    env.observation_space,
    env.action_space,
    params,
    name)

agent.learn(env)
