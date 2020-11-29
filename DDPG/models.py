import torch
from torch import tensor, cat, relu, tanh
import torch.nn as nn
from coordconv import AddLayers
from resnet import resnet18

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
		x = cat([state.float(), action.float()], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x


class ImageActor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange, n_kernels=32):
		#inSize = 821 for 4
		#inSize = 817 for 2
		super(ImageActor, self).__init__()
		self.nActions = nActions
		self.actionRange = actionRange

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

		self.fc = nn.Linear(inSize, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.mu = nn.Linear(hSize, nActions)

	def forward(self, img, state):
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat((x.float(), state.float()), dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))

		action = self.actionRange * tanh(self.mu(x))
		return action


class ImageCritic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, n_kernels=32):
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

	def forward(self, img, state, action):
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat([x.float(), state.float(), action.float()], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x


class ResNetActor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange, n_kernels=32):
		#inSize = 821 for 4
		#inSize = 817 for 2
		super(ResNetActor, self).__init__()
		self.nActions = nActions
		self.actionRange = actionRange

		self.addlayers = AddLayers()

		## Conv layers
		#output is [1,1000]
		self.conv = resnet18()

		self.fc = nn.Linear(inSize, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.mu = nn.Linear(hSize, nActions)

	def forward(self, img, state):
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat((x.float(), state.float()), dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))

		action = self.actionRange * tanh(self.mu(x))
		return action


class ResNetCritic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, n_kernels=32):
		super(ResNetCritic, self).__init__()

		self.addlayers = AddLayers()

		## Conv layers
		self.conv = resnet18()

		# insize + nActions = 825
		self.fc = nn.Linear(inSize + nActions, hSize)

		self.layers = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.layers.append(nn.Linear(hSize, hSize))
			self.norms.append(nn.LayerNorm(hSize))

		self.value = nn.Linear(hSize, 1)

	def forward(self, img, state, action):
		x = self.addlayers(img)
		x = self.conv(x)
		x = cat([x.float(), state.float(), action.float()], dim=-1).float()
		x = relu(self.fc(x))
		for layer, norm in zip(self.layers, self.norms):
			x = relu(layer(norm(x)))
		x = self.value(x)
		return x