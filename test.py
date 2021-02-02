from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv, DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv, DiscreteGradientTSCSEnv
from tscsRL.environments.RadiiTSCSEnv import ContinuousRadiiTSCSEnv, DiscreteRadiiTSCSEnv
from tscsRL.agents import ddpg, ddqn
from tscsRL import utils
import imageio
import torch

env = DiscreteGradientTSCSEnv(
	nCyl=10,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

env.config = torch.tensor([[-1.0500, 3.3387,  -0.0000, -1.6621, 3.1496, 3.3798, -3.9227, -3.1003, -1.8876, 0.5824, 1.0500, 3.3387, -0.0000, -3.9214, 1.8876, -2.5824, -3.1496, 3.3798, 3.9227, -3.1003]])

env.setMetric(env.config)
print(env.RMS)

# ## Name of the run we want to evaluate
# name = 'ddqn10cyl'

# path = 'results/' + name
# env_params = utils.jsonToDict(path + '/env_params.json')
# agent_params = utils.jsonToDict(path + '/agent_params.json')

# ## Change this environment to whatever one you need
# env = DiscreteGradientTSCSEnv(
# 	nCyl=env_params['nCyl'],
# 	kMax=env_params['kMax'],
# 	kMin=env_params['kMin'],
# 	nFreq=env_params['nFreq'],
# 	stepSize=env_params['stepSize'])

# ## Make sure these parameters are set from the env_params
# env.ep_len = env_params['ep_len']
# env.grid_size = env_params['grid_size']

# ## Change this to the correct agent you want to evaluate
# agent = ddqn.DDQNAgent(
# 	env.observation_space,
# 	env.action_space,
# 	agent_params,
# 	name)

# ## Set exploration rate to low amount
# agent.epsilon = 0.3
# # agent.noise_scale = 0.02

# ## Load weights, specify checkpoint number
# agent.load_checkpoint(path + '/checkpoints/', 5800)

# ## For creating a video of the episode
# # writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)

# ## THIS WHOLE BLOCK IS THE INTERACTION LOOP

# ## Starting from a random config
# state = env.reset()
# ## End starting from random config

# ## Starting from a predefined config
# # env.config = torch.tensor([[ 0.1844, -1.7312, -0.5715,  4.8403, -1.4565, -3.0512,  1.0017,  1.5929,
# #           0.6896, -4.1387, -2.6082,  0.7608,  4.3926, -0.0193, -2.6314, -4.8552,
# #           2.2191, -2.5639,  3.0940,  2.5856]])
# # env.counter = torch.tensor([[0.0]])
# # env.setMetric(env.config)

# # env.info['initial'] = env.RMS.item()
# # env.info['lowest'] = env.info['initial']
# # env.info['final'] = None
# # env.info['score'] = 0
# # state = env.getState()
# ## End starting from random config

# env.renderIMG(env.config)

# done = False

# results = {
# 		'config': [],
# 		'rms': [],
# 		'tscs': []}

# while not done:
# 	results['config'].append(env.config)
# 	results['rms'].append(env.RMS)
# 	results['tscs'].append(env.TSCS)

# 	# img = env.getIMG(env.radii)
# 	# writer.append_data(img.view(env.img_dim).numpy())

# 	action = agent.select_action(state)
# 	nextState, reward, done, info = env.step(action)

# 	print(env.RMS.item(), done)
# 	state = nextState

# ## Initial stuff
# initialConfig = results['config'][0]
# initialRMS = results['rms'][0]
# initialTSCS = results['tscs'][0]

# ## Optimal stuff
# minIdx = results['rms'].index(min(results['rms']))
# optimalConfig = results['config'][minIdx]
# optimalRMS = results['rms'][minIdx]
# optimalTSCS = results['tscs'][minIdx]

# print('RESULTS:')
# print(f'Initial config: {initialConfig}')
# print(f'Initial RMS: {initialRMS}')
# print(f'Initial TSCS: {initialTSCS}')
# print()
# print(f'Min config: {optimalConfig}')
# print(f'Min rms: {optimalRMS}')
# print(f'Min tscs: {optimalTSCS}')

# # writer.close()
