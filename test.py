from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv, DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv, DiscreteGradientTSCSEnv
from tscsRL.agents import ddpg, ddqn
from tscsRL import utils
import imageio
import torch

## Name of the run we want to evaluate
name = 'factor_10'

path = 'results/' + name
env_params = utils.jsonToDict(path + '/env_params.json')
agent_params = utils.jsonToDict(path + '/agent_params.json')

## Change this environment to whatever one you need
# env = DiscreteTSCSEnv(
# 	nCyl=env_params['nCyl'],
# 	kMax=env_params['kMax'],
# 	kMin=env_params['kMin'],
# 	nFreq=env_params['nFreq'],
# 	stepSize=env_params['stepSize'])

env = ContinuousGradientTSCSEnv(
	nCyl=10,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

## Make sure these parameters are set from the env_params
env.ep_len = env_params['ep_len']
env.grid_size = env_params['grid_size']
# env.grid_size = 10.0

## Change this to the correct agent you want to evaluate
# agent = ddqn.DDQNAgent(
# 	env.observation_space,
# 	env.action_space,
# 	agent_params,
# 	name)

agent = ddpg.DDPGAgent(
	env.observation_space,
	env.action_space,
	agent_params,
	name)

## Set exploration rate to low amount
#agent.epsilon = 0.05
agent.noise_scale = 0.02

## Load weights, specify checkpoint number
agent.load_checkpoint(path + '/checkpoints/', 5999)

## For creating a video of the episode
writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)


## THIS WHOLE BLOCK IS THE INTERACTION LOOP

## Starting from a random config
# state = env.reset()
## End starting from random config

## Starting from a predefined config
env.config = torch.tensor([[ 0.7213,  2.6523, -2.5113, -4.8981,  2.1382,  4.2408, -4.3452,  2.2774,
         -0.5171, -2.6907,  3.7242, -3.0200,  4.8546,  3.8931, -1.9612,  4.9669,
         -2.3093, -1.0165,  4.7938, -0.8607]])
env.counter = torch.tensor([[0.0]])
env.setMetric(env.config)

env.info['initial'] = env.RMS.item()
env.info['lowest'] = env.info['initial']
env.info['final'] = None
env.info['score'] = 0
state = env.getState()
## End starting from random config

done = False

results = {
		'config': [],
		'rms': [],
		'tscs': []}

while not done:
	results['config'].append(env.config)
	results['rms'].append(env.RMS)
	results['tscs'].append(env.TSCS)

	img = env.getIMG(env.config)
	writer.append_data(img.view(env.img_dim).numpy())

	action = agent.select_action(state)
	# action = env.action_space.sample()
	nextState, reward, done, info, invalid = env.step(action)

	print(reward, done)
	state = nextState

## Initial stuff
initialConfig = results['config'][0]
initialRMS = results['rms'][0]
initialTSCS = results['tscs'][0]

## Optimal stuff
minIdx = results['rms'].index(min(results['rms']))
optimalConfig = results['config'][minIdx]
optimalRMS = results['rms'][minIdx]
optimalTSCS = results['tscs'][minIdx]

print('RESULTS:')
print(f'Initial config: {initialConfig}')
print(f'Initial RMS: {initialRMS}')
print(f'Initial TSCS: {initialTSCS}')
print()
print(f'Min config: {optimalConfig}')
print(f'Min rms: {optimalRMS}')
print(f'Min tscs: {optimalTSCS}')

writer.close()
