import torch
from torch import relu
import torch.nn as nn
from torch import optim

class CylinderNet(nn.Module):
	def __init__(self, inSize, hSize, nHidden, nActions, lr):
		super(CylinderNet, self).__init__()
		self.fc = nn.Linear(inSize, hSize)
		self.hidden = nn.ModuleList()
		for _ in range(nHidden):
			self.hidden.append(nn.Linear(hSize, hSize))
		self.v = nn.Linear(hSize, 1)
		self.adv = nn.Linear(hSize, nActions)

		# self.opt = optim.Adam(self.parameters(), lr=lr)
		self.opt = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

		self.to(self.device)

	def forward(self, x):
		x = x.to(self.device)

		x = relu(self.fc(x))
		for layer in self.hidden:
			x = relu(layer(x))
			
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q