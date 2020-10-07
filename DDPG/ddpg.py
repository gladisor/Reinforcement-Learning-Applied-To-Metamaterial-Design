from models import Actor, Critic
import torch
from torch import tensor, cat, tanh
from torch.optim import Adam
import torch.nn.functional as F
from collections import namedtuple
from memory import NaivePrioritizedBuffer
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from env import TSCSEnv
import wandb

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

		## Networks
		self.actor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
		self.targetActor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
		self.critic = Critic(inSize, criticNHidden, criticHSize, nActions)
		self.targetCritic = Critic(inSize, criticNHidden, criticHSize, nActions)

		self.actor, self.targetActor = self.actor.cuda(), self.targetActor.cuda()
		self.critic, self.targetCritic = self.critic.cuda(), self.targetCritic.cuda()
		## Define the optimizers for both networks
		self.actorOpt = Adam(self.actor.parameters(), lr=actorLR)
		self.criticOpt = Adam(self.critic.parameters(), lr=criticLR, weight_decay=criticWD)

		## Hard update
		self.targetActor.load_state_dict(self.actor.state_dict())
		self.targetCritic.load_state_dict(self.critic.state_dict())

		## Various hyperparameters
		self.gamma = gamma
		self.tau = tau
		self.epsilon = epsilon
		self.epsDecay = epsDecay
		self.epsEnd = epsEnd

		## Transition tuple to store experience
		self.Transition = namedtuple(
			'Transition',
			('s','a','r','s_','done'))

		## Allocate memory for replay buffer and set batch size
		self.memory = NaivePrioritizedBuffer(memSize)
		self.batchSize = batchSize

		self.numEpisodes = numEpisodes
		self.epLen = epLen
		self.saveModels = 1000

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, self.epsilon, self.nActions)
			action = self.targetActor(state.cuda()).cpu() + noise
			action.clamp_(-self.actionRange, self.actionRange)
		return action

	def select_action_ou(self, state):
		pass

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
			s, a, r, s_, done = s.cuda(), a.cuda(), r.cuda(), s_.cuda(), done.cuda()
			weights = tensor([weights]).cuda()

			## Compute target
			maxQ = self.targetCritic(s_, self.targetActor(s_).detach())
			target_q = r + (1.0 - done) * self.gamma * maxQ

			## Update the critic network
			self.criticOpt.zero_grad()
			current_q = self.critic(s, a)
			criticLoss = weights @ F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')
			criticLoss.backward()
			self.criticOpt.step()

			## Update the actor network
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
		## Create file to store run data in using tensorboard

		for episode in range(self.numEpisodes):

			## Reset environment to starting state
			state = env.reset()
			episode_reward = 0

			## Log initial scattering at beginning of episode
			initial = env.RMS.item()
			lowest = initial

			for t in tqdm(range(self.epLen)):

				## Select action and observe next state, reward
				action = self.select_action(state)
				nextState, reward = env.step(action)
				episode_reward += reward

				# Update current lowest scatter
				current = env.RMS.item()
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
				self.memory.push(self.Transition(state, action, reward, nextState, done))

				## Preform bellman update
				td = self.optimize_model()

				## Break out of loop if terminal state
				if done == 1:
					break

				state = nextState

			## Print episode statistics to console
			print(
				f'#:{episode}, ' \
				f'I:{round(initial, 2)}, ' \
				f'Lowest:{round(lowest, 2)}, ' \
				f'F:{round(current, 2)}, '\
				f'Score:{round(episode_reward, 2)}, ' \
				f'td:{round(td, 2)}, ' \
				f'Epsilon: {round(self.epsilon, 2)}')

			wandb.log({'epsilon':self.epsilon, 'lowest':lowest, 'score':episode_reward})

			## Save models
			if episode % self.saveModels == 0:
				torch.save(self.targetActor.state_dict(), 'actor.pt')
				torch.save(self.targetCritic.state_dict(), 'critic.pt')

			## Reduce exploration
			self.decay_epsilon()

if __name__ == '__main__':
	# ddpg params
	IN_SIZE 		= 21
	ACTOR_N_HIDDEN 	= 2
	ACTOR_H_SIZE 	= 128
	CRITIC_N_HIDDEN = 8
	CRITIC_H_SIZE 	= 128
	N_ACTIONS 		= 8
	ACTION_RANGE 	= 0.2
	ACTOR_LR 		= 1e-4
	CRITIC_LR 		= 1e-3
	CRITIC_WD 		= 1e-2 		## How agressively to reduce overfitting
	GAMMA 			= 0.90 		## How much to value future reward
	TAU 			= 0.001 	## How much to update target network every step
	EPSILON 		= 0.75		## Scale of random noise
	EPS_DECAY 		= 0.9998	## How slowly to reduce epsilon
	EPS_END 		= 0.02 		## Lowest epsilon allowed
	MEM_SIZE 		= 1_000_000 ## How many samples in priority queue
	MEM_ALPHA 		= 0.7 		## How much to use priority queue (0 = not at all, 1 = maximum)
	MEM_BETA 		= 0.5 		## No clue ????
	BATCH_SIZE 		= 64
	NUM_EPISODES 	= 20_000
	EP_LEN 			= 100

	agent = DDPG(
		IN_SIZE, 
		ACTOR_N_HIDDEN, 
		ACTOR_H_SIZE,
		CRITIC_N_HIDDEN, 
		CRITIC_H_SIZE, 
		N_ACTIONS, 
		ACTION_RANGE, 
		ACTOR_LR, 
		CRITIC_LR, 
		CRITIC_WD, 
		GAMMA, 
		TAU, 
		EPSILON, 
		EPS_DECAY, 
		EPS_END, 
		MEM_SIZE, 
		BATCH_SIZE, 
		NUM_EPISODES,
		EP_LEN)

	## Setting memory hyperparameters
	agent.memory.alpha = MEM_ALPHA
	agent.memory.beta = MEM_BETA

	wandb.init(project='tscs')
	wandb.config.actor_n_hidden = ACTOR_N_HIDDEN
	wandb.config.actor_h_size = ACTOR_H_SIZE
	wandb.config.critic_n_hidden = CRITIC_N_HIDDEN
	wandb.config.critic_h_size = CRITIC_H_SIZE
	wandb.config.action_range = ACTION_RANGE
	wandb.config.actor_lr = ACTOR_LR
	wandb.config.critic_lr = CRITIC_LR
	wandb.config.critic_wd = CRITIC_WD
	wandb.config.gamma = GAMMA
	wandb.config.tau = TAU
	wandb.config.epsilon = EPSILON
	wandb.config.eps_decay = EPS_DECAY
	wandb.config.eps_end = EPS_END
	wandb.config.mem_size = MEM_SIZE
	wandb.config.alpha = MEM_ALPHA
	wandb.config.beta = MEM_BETA
	wandb.config.batch_size = BATCH_SIZE

	## Create env and agent
	env = TSCSEnv()

	## Run training session
	agent.learn(env)