import torch
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
		x = torch.relu(self.fc1(s.cuda()))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

class CylinderNet(nn.Module):
	def __init__(self):
		super(CylinderNet, self).__init__()
		self.fc1 = nn.Linear(19, 100)
		self.fc2 = nn.Linear(100, 100)
		self.v = nn.Linear(100, 1)
		self.adv = nn.Linear(100, 16)

	def forward(self, s):
		x = torch.cat([s[0], s[1]], dim=-1)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

class CylinderCoordConv(nn.Module):
	def __init__(self):
		super(CylinderCoordConv, self).__init__()
		## Adding coordconv layers
		self.addlayers = AddLayers()
		## Conv layers
		self.conv1 = nn.Conv2d(6, 16, kernel_size=10, stride=2)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
		self.flat = nn.Flatten()
		## Linear layers
		self.fc1 = nn.Linear(1044, 512)
		self.fc2 = nn.Linear(512, 256)
		self.fc3 = nn.Linear(256, 100)
		self.v = nn.Linear(100, 1)
		self.adv = nn.Linear(100, 16)

	def forward(self, s):
		config, tscs, rms, img = s
		x = self.addlayers(img.cuda())
		x = torch.relu(self.conv1(x))
		x = torch.relu(self.conv2(x))
		x = torch.relu(self.conv3(x))
		x = self.flat(x)
		x = torch.cat([x, config.cuda(), tscs.cuda(), rms.cuda()], dim=-1)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		x = torch.relu(self.fc3(x))
		a = self.adv(x)
		q = self.v(x) - a + a.mean(-1, keepdim=True)
		return q

if __name__ == '__main__':		
	from env import TSCSEnv

	env = TSCSEnv()
	state = env.reset()
	config, tscs, img = state
	print(config.shape, tscs.shape, img.shape)

	q = CylinderCoordConv()
	out = q(state)
	print(out.shape)
