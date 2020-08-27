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
			memory_size, batch_size, lr):

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

		self.Transition = None

	def select_action(self, state):
		## Epsilon greedy action selection		
		if random.random() > self.eps:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qt(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(16)
		return action

	def extract_tensors(self, batch):
		s = (
			cat(batch.c),
			cat(batch.tscs), 
			cat(batch.rms), 
			cat(batch.img))

		a = cat(batch.a)
		r = cat(batch.r)

		s_ = (
			cat(batch.c_),
			cat(batch.tscs_),
			cat(batch.rms_),
			cat(batch.img_))

		done = cat(batch.done)
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
				target_q_values[~done] = r[~done] + self.gamma * maxQ[~done]
				target_q_values[done] = r[done]

			## Importance sampling weights for each td error
			# Weight for appended transition is set to 1
			weights = torch.cat([
				torch.tensor([weights]).T,
				torch.tensor([[1.0]])], dim=0)
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


if __name__ == '__main__':
	GAMMA = 0.99
	EPS = 1
	EPS_END = 0.05
	EPS_DECAY = 0.99
	TARGET_UPDATE = 1000
	MEMORY_SIZE = 10_000
	BATCH_SIZE = 32
	LR = 0.0005
	NUM_EPISODES = 300

	env = gym.make('LunarLander-v2')

	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	agent.Transition = namedtuple(
		'Transition', ('s','a','r','s_','done'))

	step = 0
	running_reward = 0
	hist = []
	for episode in range(NUM_EPISODES):

		episode_reward = 0
		state = tensor([env.reset()]).float()

		for t in tqdm(range(1000)):

			action = agent.select_action(state)
			nextState, reward, done, _ = env.step(action)

			episode_reward += reward
			step += 1

			action = tensor([[action]])
			reward = tensor([[reward]]).float()
			nextState = tensor([nextState]).float()
			done = tensor([done])

			e = agent.Transition(state, action, reward, nextState, done)
			agent.memory.push(e)
			loss = agent.optimize_model(e)

			state = nextState
			if step % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())
				step = 0

			if done:
				break

		print(f'#: {episode}, Score: {round(running_reward,2)}, Eps: {round(agent.eps, 2)}')
		running_reward = running_reward*0 + episode_reward*1
		hist.append(running_reward)
		agent.finish_episode()
	
	del agent.memory.memory[:]
	env.close()
	plt.plot(hist)
	plt.show()

	torch.save(agent.Qt.state_dict(), 'm.pt')
