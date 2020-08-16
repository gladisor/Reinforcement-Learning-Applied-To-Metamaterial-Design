import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, relu
import random
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from torch.optim.lr_scheduler import StepLR

class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.idx = 0

	def push(self, transition):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.idx] = transition
		self.idx = (self.idx + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

	def __len__(self):
		return len(self.memory)

class DQN(nn.Module):
	def __init__(self):
		super(DQN, self).__init__()
		self.fc1 = nn.Linear(8, 512)
		self.bn1 = nn.BatchNorm1d(512)
		self.fc2 = nn.Linear(512, 256)
		self.bn2 = nn.BatchNorm1d(256)
		self.fc3 = nn.Linear(256, 4)

	def forward(self, x):
		x = relu(self.fc1(x))
		x = relu(self.fc2(x))
		x = self.fc3(x)
		return x

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
		self.memory = ReplayMemory(memory_size)
		self.batch_size = batch_size
		self.opt = torch.optim.Adam(self.Qp.parameters(), lr=lr)
		# self.scheduler = StepLR(self.opt, step_size=3, gamma=0.95)


	def select_action(self, state):		
		if random.random() > self.eps:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(self.Qt(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(4)
		return action

	def optimize_model(self):
		if self.memory.can_provide_sample(self.batch_size):
			experiences = self.memory.sample(self.batch_size)
			batch = Transition(*zip(*experiences))
			non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.s_)), dtype=torch.bool)
			non_final_next_states = torch.cat([s for s in batch.s_ if s is not None])
			s = torch.cat(batch.s)
			a = torch.cat(batch.a).unsqueeze(-1)
			r = torch.cat(batch.r).unsqueeze(-1)

			current_q_values = agent.Qp(s).gather(-1, a)
			with torch.no_grad():
				next_state_values = torch.zeros(self.batch_size)
				next_state_values[non_final_mask] = agent.Qt(non_final_next_states).max(-1)[0]
				target_q_values = (next_state_values.unsqueeze(-1) * self.gamma) + r

			loss = F.smooth_l1_loss(current_q_values, target_q_values)
			self.opt.zero_grad()
			loss.backward()
			self.opt.step()

	def finish_episode(self):
		self.eps *= self.eps_decay
		self.eps = max(self.eps, self.eps_end)
		# self.scheduler.step()

Transition = namedtuple('Transition', ('s','a','r','s_'))

if __name__ == '__main__':
	GAMMA = 0.99
	EPS = 1
	EPS_END = 0.01
	EPS_DECAY = 0.995
	TARGET_UPDATE = 10
	MEMORY_SIZE = 1_000_000
	BATCH_SIZE = 64
	LR = 0.001
	NUM_EPISODES = 600

	env = gym.make('LunarLander-v2')

	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	hist = []
	for episode in range(NUM_EPISODES):
		episode_reward = 0
		state = tensor([env.reset()]).float()
		for t in range(1000):
			if episode % 100 == 0:
				env.render()
			action = agent.select_action(state)
			nextState, reward, done, _ = env.step(action)
			episode_reward += reward

			action = tensor([action])
			reward = tensor([reward]).float()
			nextState = tensor([nextState]).float()

			if done:
				agent.memory.push(Transition(state, action, reward, None))
				rate = agent.opt.param_groups[0]['lr']
				print(f'#: {episode}, Score: {episode_reward}, Epsilon: {agent.eps}, LR: {rate}')
				hist.append(episode_reward)
				break

			
			agent.memory.push(Transition(state, action, reward, nextState))
			state = nextState

			agent.optimize_model()

			if t % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())

		if episode_reward > 250:
			break

		agent.finish_episode()

	env.close()
	plt.plot(hist)
	plt.show()

	torch.save(agent.Qt.state_dict(), 'model.pt')
