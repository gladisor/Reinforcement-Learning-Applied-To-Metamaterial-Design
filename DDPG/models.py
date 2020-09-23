import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions):
		super(Actor, self).__init__()
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
		x = 0.2 * tanh(self.mu(x))
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

	env = TSCSEnv()
	actor = Actor(21, 1, 64, 8)
	critic = Critic(21, 1, 64, 8)
	print(actor)
	print(critic)
	state, rms = env.reset()

	for t in range(100):
		action = actor(state)
		nextState, rms, reward, done = env.step(action)
		print(reward)
		state = nextState
