from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv, DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv, DiscreteGradientTSCSEnv
from tscsRL.agents import ddpg, ddqn
from tscsRL import utils
import imageio

name = 'ddqn4cyl0.45-0.35-8000decay'
path = 'results/' + name
env_params = utils.jsonToDict(path + '/env_params.json')
agent_params = utils.jsonToDict(path + '/agent_params.json')

# env = ContinuousGradientTSCSEnv(
# 	nCyl=env_params['nCyl'],
# 	kMax=env_params['kMax'],
# 	kMin=env_params['kMin'],
# 	nFreq=env_params['nFreq'],
# 	stepSize=env_params['stepSize'])

# agent = ddpg.DDPGAgent(
# 	env.observation_space, 
# 	env.action_space,
# 	agent_params,
# 	name)
# agent.noise_scale = 0.02

env = DiscreteTSCSEnv(
	nCyl=env_params['nCyl'],
	kMax=env_params['kMax'],
	kMin=env_params['kMin'],
	nFreq=env_params['nFreq'],
	stepSize=env_params['stepSize'])

agent = ddqn.DDQNAgent(
	env.observation_space,
	env.action_space,
	agent_params,
	name)
agent.epsilon = 0.1

agent.load_checkpoint(path + '/checkpoints/', 8000)

writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)

state = env.reset()
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

minIdx = results['rms'].index(min(results['rms']))
initialRMS = results['rms'][0]
optimalConfig = results['config'][minIdx]
optimalRMS = results['rms'][minIdx]
optimalTSCS = results['tscs'][minIdx]

print(f'Initial: {initialRMS}')
print(f'Min config: {optimalConfig}')
print(f'Min rms: {optimalRMS}')
print(f'Min tscs: {optimalTSCS}')

writer.close()
