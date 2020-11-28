from models import CylinderNet
from copy import deepcopy
from memory import NaivePrioritizedBuffer
from collections import namedtuple
import numpy as np
import torch
from torch import cat
import torch.nn.functional as F

class Agent():
	"""docstring for Agent"""
	def __init__(self, params):
		self.Qp = CylinderNet(
			params['inSize'],
			params['hSize'],
			params['nHidden'],
			params['nActions'],
			params['lr'])

		self.Qt = deepcopy(self.Qp)

		self.gamma = params['gamma']
		self.eps = 1.0
		self.epsStart = self.eps
		self.epsEnd = params['epsEnd']
		self.epsDecaySteps = params['epsDecaySteps']
		self.memory = NaivePrioritizedBuffer(params['memorySize'])
		self.batchSize = params['batchSize']
		## Transition tuple to store experience
		self.Transition = namedtuple(
			'Transition',
			('s','a','r','s_','done'))
		self.nActions = params['nActions']
		self.tau = params['tau']

		self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	def select_action(self, state):
		if np.random.random() > self.eps:
			with torch.no_grad():
				action = torch.argmax(self.Qt(state), dim=-1).item()
		else:
			action = np.random.randint(self.nActions)
		return action

	def decay_epsilon(self):
		self.eps -= (self.epsStart - self.epsEnd) / self.epsDecaySteps
		self.eps = max(self.eps, self.epsEnd)

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	def extract_tensors(self, batch):
		batch = self.Transition(*zip(*batch))
		s = cat(batch.s)
		a = cat(batch.a)
		r = cat(batch.r)
		s_ = cat(batch.s_)
		done = cat(batch.done)
		return s, a, r, s_, done

	def optimize_model(self):
		"""
		Bellman update with prioritized sampling
		"""
		if self.memory.can_provide_sample(self.batchSize):
			## Get sample from priority queue
			experiences, indices, weights = self.memory.sample(self.batchSize)

			## Convert list of transitions to tensors
			s, a, r, s_, done = self.extract_tensors(experiences)
			a = a.to(self.device)
			r = r.to(self.device)
			done = done.to(self.device)

			## Importance sampling weights
			weights = torch.tensor([weights]).T

			## Current and target Q values
			current_q_values = self.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = self.Qt(s_).max(-1, keepdim=True)[0]
				
				weights = weights.to(self.device)

				target_q_values = r + (1 - done) * self.gamma * maxQ
				td = torch.abs(target_q_values - current_q_values).detach()

			## Calculate loss and backprop
			prios = weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
			loss = prios.mean()
			self.Qp.opt.zero_grad()
			loss.backward()
			self.Qp.opt.step()

			## Update priorities of sampled batch
			self.memory.update_priorities(indices, td)

			self.soft_update(self.Qt, self.Qp)

if __name__ == '__main__':
	params = {
		'hSize': 128,
		'nHidden': 2,
		'nActions': 16,
		'lr': 1e-3,
		'gamma': 0.90,
		'epsEnd': 0.10,
		'epsDecaySteps': 10_000,
		'memorySize': 100_000,
		'batchSize': 64,
		'tau': 0.001
		}

	agent = Agent(params)