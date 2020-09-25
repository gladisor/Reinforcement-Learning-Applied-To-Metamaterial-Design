import torch
import torch.nn as nn
from torch import cat, relu, tanh
import torch.nn.functional as F
from torch.autograd import Variable
import math

## https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init
        
        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))
        
        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x
        
class NoisyActor(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions, actionRange):
		super(NoisyActor, self).__init__()
		self.actionRange = actionRange
		self.fc = NoisyLinear(inSize, hSize)
		self.hidden = nn.ModuleList()
		self.norms = nn.ModuleList()

		for _ in range(nHidden):
			self.norms.append(nn.LayerNorm(hSize))
			self.hidden.append(NoisyLinear(hSize, hSize))

		self.mu = NoisyLinear(hSize, nActions)

	def forward(self, state):
		x = relu(self.fc(state))
		for norm, layer in zip(self.norms, self.hidden):
			x = relu(layer(norm(x)))

		x = self.actionRange * tanh(self.mu(x))
		return x

	def reset_noise(self):
		self.fc.reset_noise()
		for layer in self.hidden:
			layer.reset_noise()
		self.mu.reset_noise()

class NoisyCritic(nn.Module):
	def __init__(self, inSize, nHidden, hSize, nActions):
		super(NoisyCritic, self).__init__()
		self.fc = NoisyLinear(inSize + nActions, hSize)
		self.hidden = nn.ModuleList()
		self.norms = nn.ModuleList()
		for _ in range(nHidden):
			self.norms.append(nn.LayerNorm(hSize))
			self.hidden.append(NoisyLinear(hSize, hSize))

		self.value = NoisyLinear(hSize, 1)

	def forward(self, state, action):
		x = cat([state, action], dim=-1)
		x = relu(self.fc(x))
		for norm, layer in zip(self.norms, self.hidden):
			x = relu(layer(norm(x)))
		v = self.value(x)
		return v

	def reset_noise(self):
		self.fc.reset_noise()
		for layer in self.hidden:
			layer.reset_noise()
		self.v.reset_noise()

if __name__ == '__main__':
	from env import TSCSEnv

	env = TSCSEnv()
	actor = NoisyActor(21, 1, 64, 8, 0.2)
	critic = NoisyCritic(21, 1, 64, 8)

	state, rms = env.reset()

	print(actor)
	action = actor(state)
	print(action)
	print(critic(state, action))
