from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
from models import Actor
import torch

def evaluate_actor(actor, env):
	state = env.reset()

	# env.config = torch.tensor([[0.7640,  4.5445,  0.9684, -4.3352,  4.4436, -4.4858,  4.2287,  4.4648]])
	# env.TSCS, env.RMS = env.getMetric(env.config)
	# env.counter = torch.tensor([[0.0]])
	# time = env.getTime()
	# state = torch.cat([env.config, env.TSCS, env.RMS, time], dim=-1).float() 

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
			action = actor(state) + np.random.normal(0, 1, actor.nActions) * 0
			action.clamp_(-actor.actionRange, actor.actionRange)
		
		nextState, reward = env.step(action)
		ep_reward += reward
		# print(f'RMS: {env.RMS}, Reward: {reward}')
		state = nextState

		img = env.getIMG(env.config)
		myobj.set_data(img.view(env.img_dim, env.img_dim))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.01)
	return results

if __name__ == '__main__':
	actor = Actor(21, 2, 128, 8, 0.5)
	actor.load_state_dict(torch.load('dataSets/4cyl0.45-0.35/actor.pt'))
	env = TSCSEnv(nCyl=4, k0amax=.45, k0amin=.35, nfreq=11)

	results = evaluate_actor(actor, env)
	minIdx = results['rms'].index(min(results['rms']))
	minConfig = results['config'][minIdx]

	print(minConfig)
	print(min(results['rms']))
	print(results['tscs'][minIdx])
	# img = env.getIMG(minConfig).view(env.img_dim, env.img_dim)
	# plt.imshow(img)
	# plt.show()