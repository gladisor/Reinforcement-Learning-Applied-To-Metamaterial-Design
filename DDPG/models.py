import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange):
		super(Actor, self).__init__()
		self.actionRange = actionRange
		self.fc = nn.Linear(inSize, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.mu = nn.Linear(hSize, nActions)

	def forward(self, state):
		x = relu(self.fc(state))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.actionRange * tanh(self.mu(x))
		return x

class Critic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions):
		super(Critic, self).__init__()
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.value = nn.Linear(hSize, 1)

	def forward(self, state, action):
		x = cat([state, action], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x

if __name__ == '__main__':
	from env import TSCSEnv
	import matplotlib.pyplot as plt

	actor = Actor(21, 2, 128, 8)
	actor.load_state_dict(torch.load('savedModels/actor.pt'))
	env = TSCSEnv()
	state, rms = env.reset()

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot()

	img = env.getIMG(state[0,][0:8])
	myobj = ax.imshow(img.view(env.img_dim, env.img_dim))

	for t in range(100):
		with torch.no_grad():
			action = actor(state)
		nextState, rms, reward, done = env.step(action)
		print(rms, reward)
		state = nextState

		img = env.getIMG(state[0,][0:8])
		myobj.set_data(img.view(env.img_dim, env.img_dim))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.05)
