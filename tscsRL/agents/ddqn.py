from tscsRL.agents import BaseAgent
from tscsRL.agents.models.DQN import DQN

import torch
from torch import tensor
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import wandb
import random

def default_params():
	params = {
		'n_hidden': 1, 				## Number of layers in neural network
		'h_size': 128,				## Number of neurons per layer
		'lr': 0.001,				## Learning rate
		'eps_end': 0.05,			## Minimum percentage random action rate
		'decay_timesteps': 8000,	## How many episodes to decay learning rate over
		'target_update': 10,		## How many timesteps to take before updating target network
	}

	## Join ddqn specific params with default
	base_params = BaseAgent.default_params()
	params.update(base_params)
	return params

class DDQNAgent(BaseAgent.BaseAgent):
	"""docstring for DDQNAgent"""
	def __init__(self, observation_space, action_space, params, run_name):
		super(DDQNAgent, self).__init__(observation_space, action_space, params, run_name)
		## Defining networks

		self.Qp = DQN(
			self.observation_dim,
			self.params['h_size'],
			self.params['n_hidden'],
			self.action_space.n,
			self.params['lr'])

		self.Qt = deepcopy(self.Qp)

		## cuda:0 or cpu
		self.device = self.Qp.device

		## Epsilon
		self.epsilon = 1.0 ## Percentage of random actions to take (starts at 100%)
		self.eps_end = self.params['eps_end']
		self.eps_decay_rate = (self.epsilon - self.eps_end) / self.params['decay_timesteps']

		self.update_number = 0

	def finish_episode(self):
		self.epsilon -= self.eps_decay_rate
		self.epsilon = max(self.epsilon, self.eps_end)

	def getLogger(self, config, name):
		## Specify project name for wandb to store runs from this algorithm in
		return wandb.init(project='ddqn', config=config, name=name)

	def report(self, data, logger):
		data['epsilon'] = self.epsilon
		print(data)
		if self.params['use_wandb']:
			logger.log(data)

	def select_action(self, state):
		## Epsilon greedy action selection		
		if random.random() > self.epsilon:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qp(state), dim=-1).item()
		else:
			## Explore
			action = self.action_space.sample()
		return torch.tensor([[action]])

	def random_action(self):
		return torch.tensor([[self.action_space.sample()]])

	def optimize_model(self):
		"""
		Bellman update with prioritized sampling
		"""
		if self.memory.can_provide_sample(self.batch_size):
			## Get sample from priority queue
			batch, indices, weights = self.memory.sample(self.batch_size, self.mem_beta)

			## Convert list of transitions to tensors
			s, a, r, s_, done = self.extract_tensors(batch)
			a = a.to(self.device)
			r = r.to(self.device)
			done = done.to(self.device)

			## Importance sampling weights
			weights = torch.tensor([weights]).T.to(self.device)

			## Current and target Q values
			current_q_values = self.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = self.Qt(s_).max(-1, keepdim=True)[0]
				target_q_values = r + (1.0 - done) * self.gamma * maxQ

			## Calculate loss and backprop
			weighted_loss = weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
			loss = weighted_loss.mean()
			self.Qp.opt.zero_grad()
			loss.backward()
			self.Qp.opt.step()

			## Updating priority of transition by last absolute td error
			td = torch.abs(target_q_values - current_q_values).detach()
			self.memory.update_priorities(indices, td + 1e-5)

			## Copy over policy parameters to target net
			if self.update_number % self.params['target_update'] == 0 and self.update_number != 0:
				self.Qt.load_state_dict(self.Qp.state_dict())
				## Reset update_number so we dont get an int overflow
				self.update_number = 0
			else:
				self.update_number += 1

	def save_checkpoint(self, path, episode):
		torch.save(self.Qp.state_dict(), path + f'policy_net{episode}.pt')
		torch.save(self.Qt.state_dict(), path + f'target_net{episode}.pt')

	def load_checkpoint(self, path, episode):
		self.Qp.load_state_dict(torch.load(path + f'policy_net{episode}.pt'))
		self.Qt.load_state_dict(torch.load(path + f'target_net{episode}.pt'))