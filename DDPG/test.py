from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
from models import Actor
import torch
import imageio

def evaluate(actor, env):
	state = env.reset()
	env.config = torch.tensor([[-4.5320,  3.2800, -3.8886, -3.5087, -4.2241,  0.7828,  1.1836,  3.8953]])
	env.TSCS, env.RMS = env.getMetric(env.config)
	env.counter = torch.tensor([[0.0]])
	time = env.getTime()
	state = torch.cat([env.config, env.TSCS, env.RMS, time], dim=-1).float()

	results = {
		'config': [],
		'rms': [],
		'tscs': []}
	results['config'].append(env.config)
	results['rms'].append(env.RMS)
	results['tscs'].append(env.TSCS)

	writer = imageio.get_writer('video.mp4', format='mp4', mode='I', fps=15)

	img = env.getIMG(env.config)
	writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	for t in range(100):
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		with torch.no_grad():
			action = actor(state) + np.random.normal(0, 1, actor.nActions) * 0.0
			action.clamp_(-actor.actionRange, actor.actionRange)

		nextState, reward = env.step(action)
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		state = nextState

		img = env.getIMG(env.config)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()
	return results

if __name__ == '__main__':
	actor = Actor(21, 2, 128, 8, 0.5)
	actor.load_state_dict(torch.load('dataSets/4cyl0.45-0.35/models/actor8000.pt', map_location=torch.device('cpu')))
	env = TSCSEnv(nCyl=4, k0amax=.45, k0amin=.35, nfreq=11)

	results = evaluate(actor, env)
	minIdx = results['rms'].index(min(results['rms']))

	initialRMS = results['rms'][0]
	print(f'Initial: {initialRMS}')
	print(results['config'][minIdx])
	print(results['rms'][minIdx])
	print(results['tscs'][minIdx])
