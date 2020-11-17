from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
from models import Actor
import torch

def evaluate_actor(actor, env):
	state = env.reset()

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot()

	img = env.getIMG(env.config)
	myobj = ax.imshow(img.view(env.img_dim, env.img_dim))

	results = {
		'config': [],
		'rms': [],
		'tscs': []}

	ep_reward = 0
	for t in range(100):
		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		with torch.no_grad():
			action = actor(state) + np.random.normal(0, 0.2, actor.nActions)
			action.clamp_(-actor.actionRange, actor.actionRange)
		
		nextState, reward = env.step(action)
		ep_reward += reward
		print(f'RMS: {env.RMS}, Reward: {reward}')
		state = nextState

		img = env.getIMG(env.config)
		myobj.set_data(img.view(env.img_dim, env.img_dim))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.05)
	return results

if __name__ == '__main__':
	actor = Actor(7, 2, 128, 4, 0.5)
	# actor.load_state_dict(torch.load('linearRewardFunc0.45-0.35.pt'))
	actor.load_state_dict(torch.load('actor.pt'))

	env = TSCSEnv(nCyl=2, k0amax=.46, k0amin=.44, nfreq=1)

	results = evaluate_actor(actor, env)
	minIdx = results['rms'].index(min(results['rms']))
	minConfig = results['config'][minIdx]

	print(minConfig)
	print(min(results['rms']))
	print(results['tscs'][minIdx])
	img = env.getIMG(minConfig).view(env.img_dim, env.img_dim)
	plt.imshow(img)
	plt.show()