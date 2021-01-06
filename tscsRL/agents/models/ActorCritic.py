import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn
from tscsRL.agents.models.coordconv import AddLayers


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
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.value = nn.Linear(hSize, 1)

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

	def forward(self, state, action):
		x = cat([state.to(self.device), action.to(self.device)], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x


class ImageActor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange, lr, n_kernels=32):
		super(ImageActor, self).__init__()
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

		# Specifying action info
		self.nActions = nActions
		self.actionRange = actionRange.to(self.device)

		# Defining network architecture
		self.addlayers = AddLayers()

		self.conv = nn.Sequential(
			nn.Conv2d(3, n_kernels, kernel_size=5, stride=3),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1),
			nn.Flatten()
		)

		self.fc = nn.Linear(inSize, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.mu = nn.Linear(hSize, nActions)

		## Sending parameters to device and creating optimizer
		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, img, state):
		img = img.to(self.device)
		state = state.to(self.device)
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat((x.float(), state.float()), dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))

		action = self.actionRange * tanh(self.mu(x))
		return action


class ImageCritic(nn.Module):
	def __init__(self,  inSize, nHidden, hSize, nActions, lr, wd, n_kernels=32):
		super(ImageCritic, self).__init__()

		self.addlayers = AddLayers()

		## Conv layers
		self.conv = nn.Sequential(
			nn.Conv2d(3, n_kernels, kernel_size=5, stride=3),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1),
			nn.Flatten()
		)

		# insize + nActions = 825
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.value = nn.Linear(hSize, 1)

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)

	def forward(self, img, state, action):
		img = img.to(self.device)
		state = state.to(self.device)
		action = action.to(self.device)
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat([x.float(), state.float(), action.float()], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x