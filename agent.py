import gym
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
from models import DQN

class Agent():
	def __init__(self, 
			gamma, eps, eps_end, eps_decay, 
			memory_size, batch_size, lr, cuda=True):

		self.Qp = DQN()
		self.Qt = DQN()
		self.Qt.load_state_dict(self.Qp.state_dict())
		self.Qt.eval()

		self.gamma = gamma
		self.eps = eps
		self.eps_end = eps_end
		self.eps_decay = eps_decay
		self.memory = NaivePrioritizedBuffer(memory_size)
		self.batch_size = batch_size
		self.opt = torch.optim.RMSprop(self.Qp.parameters(), lr=lr)
		self.cuda = cuda
		self.Transition = None

	def select_action(self, state):
		## Epsilon greedy action selection		
		if random.random() > self.eps:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qt(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(4)
		return action

	def extract_tensors(self, batch):
		s = (
			cat(batch.c),
			cat(batch.tscs), 
			cat(batch.rms), 
			cat(batch.img))
		# s = cat(batch.s)

		a = cat(batch.a)
		r = cat(batch.r)

		s_ = (
			cat(batch.c_),
			cat(batch.tscs_),
			cat(batch.rms_),
			cat(batch.img_))
		# s_ = cat(batch.s_)

		done = cat(batch.done)
		if self.cuda:
			a.cuda()
			r.cuda()
			done.cuda()
		return s, a, r, s_, done

	def optimize_model(self, e):
		"""
		Bellman update with prioritized sampling
		"""
		if self.memory.can_provide_sample(self.batch_size):
			experiences, indices, weights = self.memory.sample(self.batch_size)
			experiences.append(e)
			batch = self.Transition(*zip(*experiences))

			s, a, r, s_, done = self.extract_tensors(batch)

			current_q_values = self.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = self.Qt(s_).max(-1, keepdim=True)[0]

				target_q_values = torch.zeros(self.batch_size + 1, 1)
				if self.cuda:
					target_q_values.cuda()
				target_q_values[~done] = r[~done] + self.gamma * maxQ[~done]
				target_q_values[done] = r[done]

			## Importance sampling weights for each td error
			# Weight for appended transition is set to 1
			weights = torch.cat([
				torch.tensor([weights]).T,
				torch.tensor([[1.0]])])
			if self.cuda:
				weights.cuda()

			prios = weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')

			loss = prios.mean()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()

			self.memory.update_priorities(indices, prios)
			return loss.item()

	def finish_episode(self):
		"""
		Used to update hyperparameters every episode
		"""
		self.eps *= self.eps_decay
		self.eps = max(self.eps, self.eps_end)