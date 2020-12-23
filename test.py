from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg
from tscsRL import utils
import imageio

# name = 'ddpg4cyl0.45-0.35-8000decay'
name = 'ddpgGradient4cyl0.45-0.35-8000decay'

path = 'results/' + name
env_params = utils.jsonToDict(path + '/env_params.json')
agent_params = utils.jsonToDict(path + '/agent_params.json')

env = ContinuousGradientTSCSEnv(
	nCyl=env_params['nCyl'],
	kMax=env_params['kMax'],
	kMin=env_params['kMin'],
	nFreq=env_params['nFreq'],
	stepSize=env_params['stepSize'])

agent_params['noise_scale'] = 0.02

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space,
	agent_params,
	name)

agent.load_checkpoint(path + '/checkpoints/', 8000)
print(agent.noise_scale)

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
