import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange):
		super(Actor, self).__init__()
		self.nActions = nActions
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
			
		action = self.actionRange * tanh(self.mu(x))
		return action

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