import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, relu

import random
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np

class NaivePrioritizedBuffer(object):
	def __init__(self, capacity, prob_alpha=0.6):
		self.prob_alpha = prob_alpha
		self.capacity = capacity
		self.memory = []
		self.idx = 0
		self.priorities = np.ones((capacity,), dtype=np.float32)

	def push(self, transition):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.idx] = transition
		self.idx = (self.idx + 1) % self.capacity

	def sample(self, batch_size, beta=0.4):
		if len(self.memory) == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.idx]

		probs = prios ** self.prob_alpha
		probs /= probs.sum()

		indices = np.random.choice(len(self.memory), batch_size, p=probs)
		samples = [self.memory[idx] for idx in indices]

		total = len(self.memory)
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)
		return samples, indices, weights

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(8, 100)
		self.v = nn.Linear(100, 1)
		self.adv = nn.Linear(100, 4)

	def forward(self, x):
		x = relu(self.fc1(x))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

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

	def select_action(self, state):		
		if random.random() > self.eps:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qt(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(4)
		return action

	def optimize_model(self, e):
		if self.memory.can_provide_sample(self.batch_size):
			experiences, indices, weights = self.memory.sample(self.batch_size)
			experiences.append(e)
			batch = Transition(*zip(*experiences))

			s = torch.cat(batch.s)
			a = torch.cat(batch.a).unsqueeze(-1)
			r = torch.cat(batch.r).unsqueeze(-1)
			s_ = torch.cat(batch.s_)
			terminal = torch.cat(batch.terminal)

			current_q_values = agent.Qp(s).gather(-1, a)
			with torch.no_grad():
				maxQ = agent.Qt(s_).max(-1, keepdim=True)[0]

				target_q_values = torch.zeros(s_.shape[0], 1)
				target_q_values[~terminal] = r[~terminal] + self.gamma * maxQ[~terminal]
				target_q_values[terminal] = r[terminal]

			weights = torch.cat([torch.tensor([weights]).T, torch.tensor([[1.0]])], dim=0)
			prios = weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')

			loss = prios.mean()
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()

			self.memory.update_priorities(indices, prios)
			return loss.item()

	def finish_episode(self):
		self.eps *= self.eps_decay
		self.eps = max(self.eps, self.eps_end)

Transition = namedtuple('Transition', ('s','a','r','s_','terminal'))

if __name__ == '__main__':
	GAMMA = 0.99
	EPS = 1
	EPS_END = 0.05
	EPS_DECAY = 0.99
	TARGET_UPDATE = 500
	MEMORY_SIZE = 10_000
	BATCH_SIZE = 32
	LR = 0.0005
	NUM_EPISODES = 300

	env = gym.make('LunarLander-v2')

	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	from tqdm import tqdm
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

			action = tensor([action])
			reward = tensor([reward]).float()
			nextState = tensor([nextState]).float()
			done = tensor([done])
			e = Transition(state, action, reward, nextState, done)
			agent.memory.push(e)
			loss = agent.optimize_model(e)

			state = nextState
			if step % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())

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
