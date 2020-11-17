import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.distributions.normal import Normal
import numpy as np

INIT_W = 3e-3

class Actor(nn.Module):
	"""docstring for Actor"""
	def __init__(self, lr, inSize, fc, nActions, actionRange):
		super(Actor, self).__init__()
		self.fc = fc
		self.fc.insert(0, inSize)
		self.actionRange = actionRange
		self.epsilon = 1e-6

		self.layers = nn.ModuleList()
		for i in range(len(fc) - 1):
			self.layers.append(nn.Linear(self.fc[i], self.fc[i + 1]))

		self.mu = nn.Linear(self.fc[-1], nActions)
		self.mu.weight.data.uniform_(-INIT_W, INIT_W)
		self.mu.bias.data.uniform_(-INIT_W, INIT_W)

		self.sigma = nn.Linear(self.fc[-1], nActions)
		self.sigma.weight.data.uniform_(-INIT_W, INIT_W)
		self.sigma.bias.data.uniform_(-INIT_W, INIT_W)

		self.opt = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state):
		x = state.to(self.device)

		for layer in self.layers:
			x = T.relu(layer(x))

		mu = self.mu(x)
		log_std = self.sigma(x)

		log_std = T.clamp(log_std, min=-20, max=2)

		return mu, log_std

	def sample(self, state):
		mu, log_std = self.forward(state)
		std = log_std.exp()

		pi = Normal(mu, std)
		z = pi.rsample()
		action = T.tanh(z)

		log_prob = pi.log_prob(z) - T.log(1 - action.pow(2) + self.epsilon)
		log_prob = log_prob.sum(1, keepdim=True)

		return action, log_prob

	def save_checkpoint(self, path):
		T.save(self.state_dict(), path)

	def load_checkpoint(self, path):
		self.load_state_dict(T.load(path))

class Critic(nn.Module):
	"""docstring for Critic"""
	def __init__(self, lr, inSize, fc, nActions):
		super(Critic, self).__init__()
		self.fc = fc
		self.fc.insert(0, inSize + nActions)

		self.layers = nn.ModuleList()
		for i in range(len(fc) - 1):
			self.layers.append(nn.Linear(self.fc[i], self.fc[i + 1]))

		self.q = nn.Linear(self.fc[-1], 1)
		self.q.weight.data.uniform_(-INIT_W, INIT_W)
		self.q.bias.data.uniform_(-INIT_W, INIT_W)

		self.opt = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state, action):
		state, action = state.to(self.device), action.to(self.device)
		x = T.cat([state, action], dim=-1)
		
		for layer in self.layers:
			x = T.relu(layer(x))
		q = self.q(x)
		return q

	def save_checkpoint(self, path):
		T.save(self.state_dict(), path)

	def load_checkpoint(self, path):
		self.load_state_dict(T.load(path))

class Value(nn.Module):
	"""docstring for Value"""
	def __init__(self, lr, inSize, fc):
		super(Value, self).__init__()
		self.fc = fc
		self.fc.insert(0, inSize)

		self.layers = nn.ModuleList()
		for i in range(len(fc) - 1):
			self.layers.append(nn.Linear(self.fc[i], self.fc[i + 1]))

		self.value = nn.Linear(self.fc[-1], 1)
		self.value.weight.data.uniform_(-3e-3, 3e-3)
		self.value.bias.data.uniform_(-3e-3, 3e-3)

		self.opt = optim.Adam(self.parameters(), lr=lr)
		self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, state):
		x = state.to(self.device)

		for layer in self.layers:
			x = T.relu(layer(x))
		value = self.value(x)
		return value

	def save_checkpoint(self, path):
		T.save(self.state_dict(), path)

	def load_checkpoint(self, path):
		self.load_state_dict(T.load(path))

if __name__ == '__main__':
	observation_space = 19
	action_space = 8

	actor = Actor(
		lr=1e-4,
		inSize=observation_space,
		fc=[128, 128],
		nActions=action_space,
		actionRange=0.5)

	critic = Critic(
		lr=1e-3,
		inSize=observation_space,
		fc=[256, 256],
		nActions=action_space)

	value = Value(
		lr=1e-3,
		inSize=observation_space,
		fc=[256, 256])

	state = T.randn(1, 19)
	action, log_prob = actor.act(state)
	print(f'action {action}')
	print(f'log_prob {log_prob}')

	q = critic(state, action)
	print(f'q value {q}')

	value = value(state)
	print(f'state value {value}')

