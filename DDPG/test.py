from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
from models import Actor
import torch
import imageio

def evaluate(actor, env):
	state = env.reset()

	# M2 
	# env.config = torch.tensor([[4.5641, -2.7947,  2.8730,  0.4883]])

	# # M3
	# # old initial config
	# env.config = torch.tensor([[-1.8611, -4.3921,  0.6835, -4.3353, -4.4987, -4.3141]])

	# rigid initial config
	# env.config = torch.tensor([[1.5749, -2.6670, 0.3183, 1.4200, -3.2127, 1.1244]])

	# # M4
	# # old initial config
	env.config = torch.tensor([[2.1690, -1.1629, -2.6250,  2.1641,  3.1213,  1.5562,  0.3477,  4.4343]])
	
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

	# writer = imageio.get_writer('video.mp4', format='mp4', mode='I', fps=15)

	# img = env.getIMG(env.config)
	# writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	for t in range(100):
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		with torch.no_grad():
			action = actor(state) + np.random.normal(0, 1, actor.nActions) * 0.05
			action.clamp_(-actor.actionRange, actor.actionRange)

		nextState, reward = env.step(action)
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		state = nextState

		# img = env.getIMG(env.config)
		# writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	# writer.close()
	return results

if __name__ == '__main__':
	# actor = Actor(17, 2, 128, 4, 0.5)
	# actor.load_state_dict(torch.load('dataSets/2cyl0.45-0.35/actor.pt', map_location=torch.device('cpu')))
	# env = TSCSEnv(nCyl=2, k0amax=.45, k0amin=.35, nfreq=11)

	# actor = Actor(19, 2, 128, 6, 0.5)
	# actor.load_state_dict(torch.load('dataSets/3cyl0.45-0.35/actor.pt', map_location=torch.device('cpu')))
	# env = TSCSEnv(nCyl=3, k0amax=.45, k0amin=.35, nfreq=11)

	actor = Actor(21, 2, 128, 8, 0.5)
	actor.load_state_dict(torch.load('dataSets/4cyl0.45-0.35/actor.pt', map_location=torch.device('cpu')))
	env = TSCSEnv(nCyl=4, k0amax=.45, k0amin=.35, nfreq=11)

	results = evaluate(actor, env)
	minIdx = results['rms'].index(min(results['rms']))

	initialRMS = results['rms'][0]
	print(f'Initial: {initialRMS}')
	print(results['config'][minIdx])
	print(results['rms'][minIdx])
	print(results['tscs'][minIdx])
