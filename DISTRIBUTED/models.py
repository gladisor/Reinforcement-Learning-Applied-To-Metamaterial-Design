import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, observation_space, action_space, fc, actionRange):
		super(Actor, self).__init__()
		fc.insert(0, observation_space)
		fc.append(action_space)
		self.actionRange = actionRange

		self.layers = nn.ModuleList()
		for i in range(len(fc[:-1])):
			self.layers.append(nn.Linear(fc[i], fc[i+1]))

	def forward(self, x):
		for layer in self.layers[:-1]:
			x = torch.relu(layer(x))

		action = self.actionRange * torch.tanh(self.layers[-1](x))
		return action

class Critic(nn.Module):
	def __init__(self, observation_space, action_space, fc):
		super(Critic, self).__init__()
		fc.insert(0, observation_space + action_space)
		fc.append(1)

		self.layers = nn.ModuleList()
		for i in range(len(fc[:-1])):
			self.layers.append(nn.Linear(fc[i], fc[i+1]))

	def forward(self, state, action):
		x = torch.cat([state, action], dim=-1)
		for layer in self.layers[:-1]:
			x = torch.relu(layer(x))

		value = self.layers[-1](x)
		return value

if __name__ == '__main__':
	from env import DistributedTSCSEnv
	import time

	config = {
		'nCyl': 4,
		'k0amax': 0.45,
		'k0amin': 0.35,
		'nFreq': 11,
		'actionRange': 0.2,
		'episodeLength':100}

	env = DistributedTSCSEnv(config)
	state = env.reset()
	state = torch.tensor(state).float()
	print(state)

	actor = Actor(
		env.observation_space.shape[1],
		env.action_space.shape[1],
		[128] * 2,
		env.action_space.high.max())

	critic = Critic(
		env.observation_space.shape[1],
		env.action_space.shape[1],
		[128] * 8)

	print(actor)
	print(critic)

	action = actor(state)
	print(action)
	value = critic(state, action)
	print(value)