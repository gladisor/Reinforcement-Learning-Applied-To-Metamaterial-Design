from tscsRL.environments.ContinuousTSCSEnv import ContinuousTSCSEnv
from tscsRL.agents.ddpg import DDPGAgent, default_params
import imageio

env = ContinuousTSCSEnv(
	nCyl=2,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = default_params()
params['save_every'] = 100
params['decay_timesteps'] = 1000
params['num_episodes'] = 1500

params['epsilon'] = 0.02

name = 'test_agent'

agent = DDPGAgent(
	env.observation_space, 
	env.action_space,
	env.stepSize,
	params,
	name)

agent.load_checkpoint('results/2cylRigidDDPG/checkpoints/', 1000)

print(agent.epsilon)

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
	nextState, reward, done = env.step(action)

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
