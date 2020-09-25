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
	def __init__(self,
		inSize, actorNHidden, actorHSize, criticNHidden, criticHSize, 
		nActions, actionRange, actorLR, criticLR, criticWD,
		gamma, tau, epsilon, epsDecay, epsEnd,
		memSize, batchSize, numEpisodes, epLen):

		super(DDPG, self).__init__()
		## Actions
		self.nActions = nActions
		self.actionRange = actionRange

		self.actor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
		self.targetActor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
		self.critic = Critic(inSize, criticNHidden, criticHSize, nActions)
		self.targetCritic = Critic(inSize, criticNHidden, criticHSize, nActions)

		# Define the optimizers for both networks
		self.actorOpt = Adam(self.actor.parameters(), lr=actorLR)
		self.criticOpt = Adam(self.critic.parameters(), lr=criticLR, weight_decay=criticWD)

		self.targetActor.load_state_dict(self.actor.state_dict())
		self.targetCritic.load_state_dict(self.critic.state_dict())

		self.gamma = gamma
		self.tau = tau
		self.epsilon = epsilon
		self.epsDecay = epsDecay
		self.epsEnd = epsEnd

		self.Transition = namedtuple(
			'Transition',
			('s','a','r','s_','done'))

		self.memory = NaivePrioritizedBuffer(memSize)
		self.batchSize = batchSize

		self.numEpisodes = numEpisodes
		self.epLen = epLen
		self.saveModels = 1000

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, 1, self.nActions)
			action = agent.targetActor(state) + noise * self.epsilon
			action.clamp_(-self.actionRange, self.actionRange)
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
		if self.memory.can_provide_sample(self.batchSize):
			## Get data from memory
			batch, indices, weights = self.memory.sample(self.batchSize)
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
			self.memory.update_priorities(indices, td + 1e-5)
			return td.mean().item()

	def decay_epsilon(self):
		self.epsilon *= self.epsDecay
		self.epsilon = max(self.epsilon, self.epsEnd)

	def learn(self, env):
		writer = SummaryWriter('runs/ddpg')

		for episode in range(self.numEpisodes):
			state, rms = env.reset()
			episode_reward = 0

			initial = rms.item()
			lowest = initial

			for t in tqdm(range(self.epLen)):
				action = self.select_action(state)
				nextState, rms, reward, done = env.step(action)
				episode_reward += reward

				# Update current lowest
				current = rms.item()
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
				e = self.Transition(state, action, reward, nextState, done)
				self.memory.push(e)

				## Preform bellman update
				td = self.optimize_model()

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
				f'Epsilon: {round(self.epsilon, 2)}')

			writer.add_scalar('train/score', episode_reward, episode)
			writer.add_scalar('train/lowest', lowest, episode)

			if episode % self.saveModels == 0:
				torch.save(self.actor.state_dict(), 'actor.pt')
				torch.save(self.critic.state_dict(), 'critic.pt')

			self.decay_epsilon()


if __name__ == '__main__':
	# ddpg params
	IN_SIZE = 21
	ACTOR_N_HIDDEN = 2
	ACTOR_H_SIZE = 128
	CRITIC_N_HIDDEN = 6
	CRITIC_H_SIZE = 128
	N_ACTIONS = 8
	ACTION_RANGE = 0.2
	ACTOR_LR = 1e-4
	CRITIC_LR = 1e-3
	CRITIC_WD = 1e-2
	GAMMA = 0.99
	TAU = 0.001
	EPSILON = 0.75
	EPS_DECAY = 0.9998
	EPS_END = 0.05
	MEM_SIZE = 300_000
	BATCH_SIZE = 64
	NUM_EPISODES = 30_000
	EP_LEN = 100

	agent = DDPG(
		IN_SIZE, ACTOR_N_HIDDEN, ACTOR_H_SIZE,
		CRITIC_N_HIDDEN, CRITIC_H_SIZE, N_ACTIONS, 
		ACTION_RANGE, ACTOR_LR, CRITIC_LR, CRITIC_WD, 
		GAMMA, TAU, EPSILON, EPS_DECAY, EPS_END, MEM_SIZE, 
		BATCH_SIZE, NUM_EPISODES,EP_LEN)

	## Create env and agent
	env = TSCSEnv()

	## Run training session
	agent.learn(env)