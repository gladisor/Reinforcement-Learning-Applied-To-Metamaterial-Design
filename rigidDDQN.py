from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import DiscreteGradientTSCSEnv
from tscsRL.agents import ddqn
from tscsRL import utils
import matplotlib.pyplot as plt

name = 'ddqn10cyl_differentReward'

# path = 'results/' + name
# env_params = utils.jsonToDict(path + '/env_params.json')
# agent_params = utils.jsonToDict(path + '/agent_params.json')

env = DiscreteGradientTSCSEnv(
	nCyl=10,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.50)

env.grid_size = env.grid_size
env.ep_len = env.ep_len

# print(env.grid_size)
# state = env.reset()
# img = env.getIMG(env.config)
# plt.imshow(img.view(env.img_dim))
# plt.show()

params = ddqn.default_params()
params['batch_size'] = 256
params['decay_timesteps'] = 8000
params['num_episodes'] = 8500
params['save_every'] = 100
params['use_wandb'] = True

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	params, 
	name)

agent.learn(env)