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
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q
		
class CylinderCoordConv(nn.Module):
	def __init__(self, useCuda):
		super(CylinderCoordConv, self).__init__()
		self.useCuda = useCuda
		## Adding coordconv layers
		self.addlayers = AddLayers(self.useCuda)
		## Conv layers
		self.conv1 = nn.Conv2d(3, 8, kernel_size=10, stride=2)
		self.conv2 = nn.Conv2d(8, 16, kernel_size=10, stride=2)
		self.flat = nn.Flatten()
		## Linear layers
		self.fc1 = nn.Linear(597, 256)
		self.fc2 = nn.Linear(256, 128)
		self.v = nn.Linear(128, 1)
		self.adv = nn.Linear(128, 16)

	def forward(self, s):
		config, tscs, rms, img, time = s
		if self.useCuda:
			config = config.cuda()
			tscs = tscs.cuda()
			rms = rms.cuda()
			img = img.cuda()
			time = time.cuda()
			
		x = self.addlayers(img)
		x = relu(self.conv1(x))
		x = relu(self.conv2(x))
		x = self.flat(x)
		x = torch.cat([x, config, tscs, rms, time], dim=-1)
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

if __name__ == '__main__':		
	from env import TSCSEnv

	env = TSCSEnv()
	state = env.reset()
	config, tscs, rms, img, time = state
	print(config.shape, tscs.shape, rms.shape, img.shape, time.shape)

	q = CylinderCoordConv(useCuda=True).cuda()
	print(q)
	out = q(state)
	print(out.shape)
