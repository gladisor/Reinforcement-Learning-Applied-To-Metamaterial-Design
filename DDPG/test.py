from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
from models import Actor
import torch
import imageio

def evaluate(actor, env):
	state = env.reset()
	# env.config = torch.tensor([[ 4.5641, -2.7947,  2.8730,  0.4883]])
	# env.TSCS, env.RMS = env.getMetric(env.config)
	# env.counter = torch.tensor([[0.0]])
	# time = env.getTime()
	# state = torch.cat([env.config, env.TSCS, env.RMS, time], dim=-1).float()

	results = {
		'config': [],
		'rms': [],
		'tscs': []}
	results['config'].append(env.config)
	results['rms'].append(env.RMS)
	results['tscs'].append(env.TSCS)

	writer = imageio.get_writer('10cyl.mp4', format='mp4', mode='I', fps=15)

	img = env.getIMG(env.config)
	writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	for t in range(100):
		with torch.no_grad():
			action = np.random.uniform(
				-actor.actionRange, 
				actor.actionRange, 
				size=(1, actor.nActions))
			# action = actor(state) + np.random.normal(0, 1, actor.nActions) * 0.05
			# action.clamp_(-actor.actionRange, actor.actionRange)

		nextState, reward = env.step(action)
		print(reward)
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		state = nextState

		img = env.getIMG(env.config)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()
	return results

if __name__ == '__main__':
	from gradientExperement import contGradientEnv
	NCYL = 10
	KMAX = .45
	KMIN = .35
	NFREQ = 11

	IN_SIZE = 2 * NCYL + NFREQ + 2
	N_ACTIONS = 2 * NCYL
	ACTION_RANGE = 0.2

	env = contGradientEnv(nCyl=NCYL, k0amax=KMAX, k0amin=KMIN, nFreq=NFREQ)

	actor = Actor(env.observation_space, 2, 128, env.action_space, ACTION_RANGE)
	# actor.load_state_dict(torch.load('savedModels/gradInfo3000Decay/actor3000.pt', map_location=torch.device('cpu')))

	results = evaluate(actor, env)
	minIdx = results['rms'].index(min(results['rms']))

	initialRMS = results['rms'][0]
	optimalConfig = results['config'][minIdx]
	optimalRMS = results['rms'][minIdx]
	optimalTSCS = results['tscs'][minIdx]

	print(f'Initial: {initialRMS}')
	print(f'Min config: {optimalConfig}')
	print(f'Min rms: {optimalRMS}')
	print(f'Min tscs: {optimalTSCS}')
