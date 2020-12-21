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
		'n_hidden': 1,
		'h_size': 128,
		'lr': 0.005,
		'momentum': 0.9,
		'eps_end': 0.05,
		'decay_timesteps': 8000,
		'target_update': 10,
		'gamma': 0.9,
		'batch_size': 64
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
			observation_space,
			self.params['h_size'],
			self.params['n_hidden'],
			action_space,
			self.params['lr'])

		self.Qt = deepcopy(self.Qp)

		## cuda:0 or cpu
		self.device = self.Qp.device

		## Spaces
		self.observation_space = observation_space
		self.action_space = action_space

		## Epsilon
		self.epsilon = 1.0
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
			action = np.random.randint(self.action_space)
		return torch.tensor([[action]])

	def random_action(self):
		return torch.tensor([[np.random.randint(self.action_space)]])

	def optimize_model(self):
		"""
		Bellman update with prioritized sampling
		"""
		if self.memory.can_provide_sample(self.params['batch_size']):
			## Get sample from priority queue
			batch, indices, weights = self.memory.sample(self.params['batch_size'], self.mem_beta)

			## Convert list of transitions to tensors
			s, a, r, s_, done = self.extract_tensors(batch)
			a = a.to(self.device)

			## Importance sampling weights
			weights = torch.tensor([weights]).to(self.device)

			## Current and target Q values
			current_q_values = self.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = self.Qt(s_).max(-1, keepdim=True)[0]
				target_q_values = r.to(self.device) + (1 - done.to(self.device)) * self.params['gamma'] * maxQ

			## Calculate loss and backprop
			loss = weights @ F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
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