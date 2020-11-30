import torch
from torch import relu
import torch.nn as nn
from coordconv import AddLayers
import torch.nn.functional as F
import math
from torch.autograd import Variable

class DQN(nn.Module):
	## Base model for lunar lander
	def __init__(self):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(8, 100)
		self.v = nn.Linear(100, 1)
		self.adv = nn.Linear(100, 4)

	def forward(self, s):
		x = torch.relu(self.fc1(s))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

class CylinderNet(nn.Module):
	def __init__(self, h_size, n_hidden):
		super(CylinderNet, self).__init__()
		self.fc = nn.Linear(21, h_size)
		self.hidden = nn.ModuleList()
		for _ in range(n_hidden):
			self.hidden.append(nn.Linear(h_size, h_size))
		self.v = nn.Linear(h_size, 1)
		self.adv = nn.Linear(h_size, 16)

	def forward(self, s):
		x = torch.cat([*s], dim=-1)
		if next(self.parameters()).is_cuda:
			x = x.cuda()

		x = relu(self.fc(x))
		for layer in self.hidden:
			x = relu(layer(x))
			
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q
		
class CylinderCoordConv(nn.Module):
	def __init__(self, n_kernels, h_size, useCuda):
		super(CylinderCoordConv, self).__init__()
		## Adding coordconv layers
		self.addlayers = AddLayers()
		## Conv layers
		self.conv = nn.Sequential(
			nn.Conv2d(3, n_kernels, kernel_size=5, stride=3),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=4, stride=2),
			nn.ReLU(),
			nn.Conv2d(n_kernels, n_kernels, kernel_size=3, stride=1),
			nn.Flatten(),
			nn.Linear(n_kernels * 5 * 5, 100),
			nn.ReLU())
		## Linear layers
		self.fc = nn.Sequential(
			nn.Linear(121, h_size),
			nn.ReLU(),
			nn.Linear(h_size, h_size),
			nn.ReLU())

		self.adv = nn.Linear(h_size, 16)
		self.v = nn.Linear(h_size, 1)

	def forward(self, s):
		config, tscs, rms, img, time = s
		nums = torch.cat([config, tscs, rms, time], dim=-1)
		if next(self.parameters()).is_cuda:
			nums = nums.cuda()
			img = img.cuda()
			
		x = self.addlayers(img)

		x = relu(self.conv1(x))
		x = relu(self.conv2(x))
		x = self.flat(x)

		x = torch.cat([x, nums], dim=-1)
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))

		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

class NoisyDQN(nn.Module):
	def __init__(self):
		super(NoisyDQN, self).__init__()

		self.linear = nn.Linear(21, 128)
		self.noisy1 = NoisyLinear(128, 128)
		self.noisy2 = NoisyLinear(128, 16)

	def forward(self, s):
		x = torch.cat([*s], dim=-1)
		x = x.cuda()
		x = relu(self.linear(x))
		x = relu(self.noisy1(x))
		x = self.noisy2(x)
		return x

	def act(self, state):
		state = Variable(torch.FloatTensor(state.unsqueeze(0), volatile=True))
		q_value = self.forward(state)
		action = q_value.max(1)[1].data[0]
		return action

	def reset_noise(self):
		self.noisy1.reset_noise()
		self.noisy2.reset_noise()

class NoisyLinear(nn.Module):
	def __init__(self, in_features, out_features, std_init=0.4):
		super(NoisyLinear, self).__init__()
		
		self.in_features = in_features
		self.out_features = out_features
		self.std_init = std_init

		self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
		self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
		self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

		self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
		self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
		self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

		self.reset_parameters()
		self.reset_noise()

	def forward(self, x):
		if self.training:
			weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
			bias = self.bias_mu + self.bias_sigma.mul(Variable(self.bias_epsilon))
		else:
			weight = self.weight_mu
			bias = self.bias_mu
		
		return F.linear(x, weight, bias)
	
	def reset_parameters(self):
		mu_range = 1 / math.sqrt(self.weight_mu.size(1))

		self.weight_mu.data.uniform_(-mu_range, mu_range)
		self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

	def reset_noise(self):
		epsilon_in = self._scale_noise(self.in_features)
		epsilon_out = self._scale_noise(self.out_features)

		self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
		self.bias_epsilon.copy_(self._scale_noise(self.out_features))

	def _scale_noise(self, size):
		x = torch.randn(size)
		x = x.sign().mul(x.abs().sqrt())
		return x


if __name__ == '__main__':		
	from env import TSCSEnv

	env = TSCSEnv()
	state = env.reset()
	config, tscs, rms, img, time = state
	print(config.shape, tscs.shape, rms.shape, img.shape, time.shape)

	q = CylinderCoordConv(
		n_kernels=32, 
		h_size=128, 
		useCuda=False)

	print(q)
	out = q(state)
	print(out.shape)
