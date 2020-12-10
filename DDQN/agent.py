import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, cat

import random
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from tqdm import tqdm

from memory import NaivePrioritizedBuffer

class Agent():
	def __init__(self, 
			gamma, eps, eps_end, eps_decay_steps, 
			memory_size, batch_size, lr):

		## Networks -- Change this to custom nets
		self.Qp = None
		self.opt = None
		self.Qt = None

		## Hyperperameters
		self.gamma = gamma
		self.eps = eps
		self.eps_start = eps
		self.eps_end = eps_end
		self.eps_decay_steps = eps_decay_steps
		self.memory = NaivePrioritizedBuffer(memory_size)
		self.batch_size = batch_size
		self.Transition = None
		self.nActions = None

	def select_action(self, state):
		## Epsilon greedy action selection		
		if random.random() > self.eps:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qp(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(self.nActions)
		return action

	def extract_tensors(self, batch):
		## State info
		s = cat(batch.s)

		## Action, reward
		a = cat(batch.a).to(self.Qp.device)
		r = cat(batch.r).to(self.Qp.device)

		## Next state info
		s_ = cat(batch.s_)
		
		done = cat(batch.done).to(self.Qp.device)
		return s, a, r, s_, done

	def optimize_model(self):
		"""
		Bellman update with prioritized sampling
		"""
		if self.memory.can_provide_sample(self.batch_size):
			## Get sample from priority queue
			experiences, indices, weights = self.memory.sample(self.batch_size)
			batch = self.Transition(*zip(*experiences))

			## Convert list of transitions to tensors
			s, a, r, s_, done = self.extract_tensors(batch)

			## Importance sampling weights
			weights = torch.tensor([weights]).T

			## Current and target Q values
			current_q_values = self.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = self.Qt(s_).max(-1, keepdim=True)[0]
				target_q_values = torch.zeros(self.batch_size, 1)
				
				if next(self.Qp.parameters()).is_cuda:
					target_q_values = target_q_values.cuda()
					weights = weights.cuda()

				# target_q_values = r + (1 - done) * self.gamma * maxQ
				target_q_values[~done] = r[~done] + self.gamma * maxQ[~done]
				target_q_values[done] = r[done]
				td = target_q_values - current_q_values

			## Calculate loss and backprop
			prios = weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
			loss = prios.mean()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()

			## Update priorities of sampled batch
			self.memory.update_priorities(indices, torch.abs(td).detach())
			return loss.item()

	def decay_epsilon(self):
		self.eps -= (self.eps_start - self.eps_end) / self.eps_decay_steps
		self.eps = max(self.eps, self.eps_end)