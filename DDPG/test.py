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

	ep_reward = 0
	for t in range(100):
		with torch.no_grad():
			action = actor(state)
			
		nextState, reward = env.step(action)
		ep_reward += reward
		print(f'RMS: {env.RMS}, Reward: {reward}')
		state = nextState

		img = env.getIMG(env.config)
		myobj.set_data(img.view(env.img_dim, env.img_dim))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.05)

if __name__ == '__main__':
	actor = Actor(21, 2, 128, 8, 0.2)
	actor.load_state_dict(torch.load('savedModels/actor9000.pt'))
	env = TSCSEnv()

	evaluate_actor(actor, env)