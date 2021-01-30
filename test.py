from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv, DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv, DiscreteGradientTSCSEnv
from tscsRL.agents import ddpg, ddqn
from tscsRL import utils
import imageio
import torch

## Name of the run we want to evaluate
# name = 'ddqn_M3_ni_POR'
# name = 'ddqn_thinshell_Ni_M3'
# name = 'ddqn_thinshell_Ni_M4'
name = 'ddpg_M3_thinshell_ni_rerun'

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

# env = DiscreteGradientTSCSEnv(
# 	nCyl=env_params['nCyl'],
# 	kMax=env_params['kMax'],
# 	kMin=env_params['kMin'],
# 	nFreq=env_params['nFreq'],
# 	stepSize=env_params['stepSize'])

env = ContinuousGradientTSCSEnv(
	nCyl=env_params['nCyl'],
	kMax=env_params['kMax'],
	kMin=env_params['kMin'],
	nFreq=env_params['nFreq'],
	stepSize=env_params['stepSize'])

## Make sure these parameters are set from the env_params
env.ep_len = env_params['ep_len']
env.grid_size = env_params['grid_size']

## Change this to the correct agent you want to evaluate
# agent = ddqn.DDQNAgent(
# 	env.observation_space,
# 	env.action_space,
# 	agent_params,
# 	name)

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

## Set exploration rate to low amount
# agent.epsilon = 0.05
agent.noise_scale = 0.02

## Load weights, specify checkpoint number
agent.load_checkpoint(path + '/checkpoints/', 2000)

## For creating a video of the episode
writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)


## THIS WHOLE BLOCK IS THE INTERACTION LOOP

## Starting from a random config
# state = env.reset()
## End starting from random config

## Starting from a predefined config
# env.config = torch.tensor([[-4.9074,  3.9546,  2.6997,  0.7667,  0.6999,  4.5946,  4.9415, -0.2377]])

# env.config = torch.tensor([[ 1.5749, -2.6670,  0.3183,  1.4200, -3.2127,  1.1244]])

# M3 thin shells initial config
# env.config = torch.tensor([[-1.8611, -4.3921, 0.6835, -4.3353, -4.4987, -4.3141]])

# M3 ddqn thin shell initial, amaris
# env.config = torch.tensor([[1.5749, -2.6670, 0.3183, 1.4200, -3.2127, 1.1244]])

# M3 ddqn rigid initial config
env.config = torch.tensor([[1.5749, -2.6670, 0.3183, 1.4200, -3.2127, 1.1244]])

# M4 ddqn rigid initial config
# env.config = torch.tensor([[2.1690, -1.1629, -2.6250, 2.1641, 3.1213, 1.5562, 0.3477, 4.4343]])

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
	nextState, reward, done, info = env.step(action)

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
