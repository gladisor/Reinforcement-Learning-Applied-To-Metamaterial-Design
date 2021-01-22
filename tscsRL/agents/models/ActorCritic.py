import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn

class Actor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange, lr):
		super(Actor, self).__init__()
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		
		## Specifying action info
		self.nActions = nActions
		self.actionRange = actionRange.to(self.device)

		## Defining network archetecture
		self.fc = nn.Linear(inSize, hSize)
		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))
		self.mu = nn.Linear(hSize, nActions)

		# self.mu.weight.data.uniform_(-3e-3, 3e-3)
		# self.mu.bias.data.uniform_(-3e-3, 3e-3)

		## Sending parameters to device and creating optimizer
		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		x = x.to(self.device)
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
			
		action = self.actionRange * tanh(self.mu(x))
		return action

class Critic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, lr, wd):
		super(Critic, self).__init__()
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.value = nn.Linear(hSize, 1)
		
		# self.value.weight.data.uniform_(-3e-3, 3e-3)
		# self.value.bias.data.uniform_(-3e-3, 3e-3)

		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

	def forward(self, state, action):
		x = cat([state.to(self.device), action.to(self.device)], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x