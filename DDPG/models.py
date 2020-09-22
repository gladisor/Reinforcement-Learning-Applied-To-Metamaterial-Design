import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions):
		super(Actor, self).__init__()
		self.fc = nn.Linear(inSize, hSize)

		self.hidden = nn.ModuleList()
		for _ in range(nHidden):
			self.hidden.append(nn.Linear(hSize, hSize))

		self.mu = nn.Linear(hSize, nActions)

	def forward(self, state):
		x = relu(self.fc(state))
		for layer in self.hidden:
			x = relu(layer(x))
		x = 0.2 * tanh(self.mu(x))
		return x

class Critic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions):
		super(Critic, self).__init__()
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.hidden = nn.ModuleList()
		for _ in range(nHidden):
			self.hidden.append(nn.Linear(hSize, hSize))

		self.value = nn.Linear(hSize, 1)

	def forward(self, state, action):
		x = cat([state, action], dim=-1).float()
		x = relu(self.fc(x))
		for layer in self.hidden:
			x = relu(layer(x))
		x = self.value(x)
		return x

if __name__ == '__main__':
	import gym

	actor = Actor(3, 1, 64, 1)
	critic = Critic(3, 1, 64, 1)
	env = gym.make('Pendulum-v0')
	state = env.reset()
	state = tensor([state]).float()

	for t in range(100):
		env.render()
		action = actor(state)
		nextState, reward, done, _ = env.step([action.item()])
		print(action, reward)
		nextState = tensor([nextState]).float()
		state = nextState

	env.close()
