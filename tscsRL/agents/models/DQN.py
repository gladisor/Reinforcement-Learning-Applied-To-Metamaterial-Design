import torch
from torch import relu
import torch.nn as nn

class DQN(nn.Module):
	def __init__(self, inSize, h_size, n_hidden, nActions, lr):
		super(DQN, self).__init__()
		self.fc = nn.Linear(inSize, h_size)
		self.hidden = nn.ModuleList()
		for _ in range(n_hidden):
			self.hidden.append(nn.Linear(h_size, h_size))
		self.v = nn.Linear(h_size, 1)
		self.adv = nn.Linear(h_size, nActions)

		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.to(self.device)
		self.opt = torch.optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		x = x.to(self.device)

		x = relu(self.fc(x))
		for layer in self.hidden:
			x = relu(layer(x))
			
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q