import torch
from torch import relu
import torch.nn as nn
from coordconv import AddLayers

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
	def __init__(self, useCuda):
		super(CylinderNet, self).__init__()
		self.useCuda = useCuda
		self.fc1 = nn.Linear(21, 128)
		self.fc2 = nn.Linear(128, 128)
		self.v = nn.Linear(128, 1)
		self.adv = nn.Linear(128, 16)

	def forward(self, s):
		config, tscs, rms, time = s
		if self.useCuda:
			config = config.cuda()
			tscs = tscs.cuda()
			rms = rms.cuda()
			time = time.cuda()

		x = torch.cat([config, tscs, rms, time], dim=-1)
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q
		
class CylinderCoordConv(nn.Module):
	def __init__(self, n_kernels, h_size, useCuda):
		super(CylinderCoordConv, self).__init__()
		self.useCuda = useCuda
		## Adding coordconv layers
		self.addlayers = AddLayers(self.useCuda)
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
		if self.useCuda:
			config = config.cuda()
			tscs = tscs.cuda()
			rms = rms.cuda()
			img = img.cuda()
			time = time.cuda()
			
		x = self.addlayers(img)
		x = self.conv(x)
		x = torch.cat([x, config, tscs, rms, time], dim=-1)
		x = self.fc(x)

		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

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
