from models import Actor, Critic
import torch
from torch import tensor, cat
from torch.optim import Adam
import torch.nn.functional as F
from collections import namedtuple
from memory import NaivePrioritizedBuffer
from noise import OrnsteinUhlenbeckActionNoise
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from env import TSCSEnv

class DDPG():
	def __init__(self):
		super(DDPG, self).__init__()
		self.nActions = 8
		self.actor = Actor(21, 2, 128, self.nActions)
		self.targetActor = Actor(21, 2, 128, self.nActions)
		self.critic = Critic(21, 4, 128, self.nActions)
		self.targetCritic = Critic(21, 4, 128, self.nActions)

		# Define the optimizers for both networks
		self.actorOpt = Adam(self.actor.parameters(), lr=1e-4)
		self.criticOpt = Adam(self.critic.parameters(), lr=1e-3, weight_decay=1e-2)

		self.targetActor.load_state_dict(self.actor.state_dict())
		self.targetCritic.load_state_dict(self.critic.state_dict())

		self.gamma = 0.99
		self.tau = 0.001
		self.action_low = -0.2
		self.action_high = 0.2
		self.epsilon = 0.75
		self.eps_decay = 0.9998
		self.eps_end = 0.05

		self.Transition = namedtuple(
			'Transition',
			('s','a','r','s_','done'))

		self.memory = NaivePrioritizedBuffer(300_000)
		self.batch_size = 64

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, self.epsilon, self.nActions)
			action = agent.targetActor(state) + noise
			action.clamp_(self.action_low, self.action_high)
		return action

	def extract_tensors(self, batch):
		batch = self.Transition(*zip(*batch))
		s = cat(batch.s)
		a = cat(batch.a)
		r = cat(batch.r)
		s_ = cat(batch.s_)
		done = cat(batch.done)
		return s, a, r, s_, done

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	def optimize_model(self):
		if self.memory.can_provide_sample(self.batch_size):
			## Get data from memory
			batch, indices, weights = self.memory.sample(self.batch_size)
			s, a, r, s_, done = self.extract_tensors(batch)
			weights = tensor([weights])

			## Compute target
			maxQ = self.targetCritic(s_, self.targetActor(s_).detach())
			target_q = r + (1.0 - done) * self.gamma * maxQ

			# Update the critic network
			self.criticOpt.zero_grad()
			current_q = self.critic(s, a)
			criticLoss = weights @ F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')
			criticLoss.backward()
			self.criticOpt.step()

			# Update the actor network
			self.actorOpt.zero_grad()
			actorLoss = -self.critic(s, self.actor(s)).mean()
			actorLoss.backward()
			self.actorOpt.step()

			## Copy policy weights over to target net
			self.soft_update(self.targetActor, self.actor)
			self.soft_update(self.targetCritic, self.critic)

			## Updating priority of transition by last absolute td error
			td = torch.abs(target_q - current_q).detach()
			self.memory.update_priorities(indices, td)
			return td.mean().item()

	def decay_epsilon(self):
		self.epsilon *= self.eps_decay
		self.epsilon = max(self.epsilon, self.eps_end)


if __name__ == '__main__':
	## ddpg params
	# N_ACTIONS = 8
	# ACTOR_N_HIDDEN = 2
	# ACTOR_H_SIZE = 128
	# CRITIC_N_HIDDEN = 4
	# CRITIC_H_SIZE = 128
	# ACTOR_LR = 1e-4
	# CRITIC_LR = 1e-3
	# CRITIC_WD = 1e-2
	# GAMMA = 0.99
	# TAU = 0.001
	# ACTION_LOW = -0.2
	# ACTION_HIGH = 0.2
	# EPSILON = 0.75
	# EPS_DECAY = 0.9995
	# EPS_END = 0.05
	# MEM_SIZE = 300_000
	# BATCH_SIZE = 64
	# ## Episode hyperparams
	# NUM_EPISODES = 30_000
	# EP_LEN = 100
	# SAVE_MODELS = 1000

	## Create env and agent
	env = TSCSEnv()
	agent = DDPG()

	writer = SummaryWriter('runs/ddpg-normalDist-LargerNets')

	for episode in range(NUM_EPISODES):
		state = env.reset()
		episode_reward = 0

		initial = state[0][8:18].mean().item()
		lowest = initial

		for t in tqdm(range(EP_LEN)):
			action = agent.select_action(state)
			nextState, reward, done = env.step(action)
			episode_reward += reward

			# Update current lowest
			current = state[0][8:18].mean().item()
			if current < lowest:
				lowest = current

			## Check if terminal
			if t == EP_LEN - 1:
				done = 1
			else:
				done = 0

			## Cast reward and done as tensors
			reward = tensor([[reward]]).float()
			done = tensor([[done]])

			## Store transition in memory
			e = agent.Transition(state, action, reward, nextState, done)
			agent.memory.push(e)

			## Preform bellman update
			td = agent.optimize_model()

			if done == 1:
				break

			state = nextState

		print(
			f'#:{episode}, ' \
			f'I:{round(initial, 2)}, ' \
			f'Lowest:{round(lowest, 2)}, ' \
			f'F:{round(current, 2)}, '\
			f'Score:{round(episode_reward, 2)}, ' \
			f'Mean td:{round(td, 2)}, ' \
			f'Epsilon: {round(agent.epsilon, 2)}')

		writer.add_scalar('train/score', episode_reward, episode)
		writer.add_scalar('train/lowest', lowest, episode)

		if episode % SAVE_MODELS == 0:
			torch.save(agent.actor.state_dict(), 'actor.pt')
			torch.save(agent.critic.state_dict(), 'critic.pt')

		agent.decay_epsilon()



