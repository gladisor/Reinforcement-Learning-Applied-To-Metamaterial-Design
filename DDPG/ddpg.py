from models import Actor, Critic
import torch
from torch import tensor, cat, tanh
from torch.optim import Adam
import torch.nn.functional as F
from collections import namedtuple
from memory import NaivePrioritizedBuffer
import numpy as np
from tqdm import tqdm
from env import TSCSEnv
import wandb
from noise import OrnsteinUhlenbeckActionNoise

class DDPG():
	def __init__(self,
		inSize, actorNHidden, actorHSize, criticNHidden, criticHSize, 
		nActions, actionRange, actorLR, criticLR, criticWD,
		gamma, tau, epsilon, decay_timesteps, epsEnd,
		memSize, batchSize, numEpisodes, epLen):

		super(DDPG, self).__init__()
		## Actions
		self.observation_size = inSize
		self.nActions = nActions
		self.actionRange = actionRange
		self.noise = OrnsteinUhlenbeckActionNoise(np.zeros(self.nActions), 1)

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
		self.epsStart = epsilon
		self.epsilon = epsilon
		self.decay_timesteps = decay_timesteps
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
		self.saveEvery = 1000
		self.randomEpisodes = 0
		self.learningBegins = 0

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, 1, self.nActions) * self.epsilon
			action = self.actor(state.cuda()).cpu() + noise
			action.clamp_(-self.actionRange, self.actionRange)
		return action

	def random_uniform_action(self):
		action = np.random.uniform(
			-self.actionRange, 
			self.actionRange, 
			size=(1, self.nActions))

		action = torch.tensor(action)
		return action

	def select_action_ou(self, state):
		with torch.no_grad():
			action = self.actor(state.cuda()).cpu() + self.noise() * self.epsilon
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

	def decay_epsilon(self):
		# self.epsilon *= self.epsDecay
		self.epsilon -= (self.epsStart - self.epsEnd) / self.decay_timesteps
		self.epsilon = max(self.epsilon, self.epsEnd)

	def learn(self, env):
		## Create file to store run data in using tensorboard
		array_size = self.numEpisodes * self.epLen 
		state_array = torch.zeros(array_size, self.observation_size)
		action_array = torch.zeros(array_size, self.nActions)
		reward_array = torch.zeros(array_size, 1)
		next_state_array = torch.zeros(array_size, self.observation_size)
		done_array = torch.zeros(array_size, 1)
		array_index = 0

		for episode in range(self.numEpisodes):

			## Reset environment to starting state
			state = env.reset()
			episode_reward = 0

			## Log initial scattering at beginning of episode
			initial = env.RMS.item()
			lowest = initial

			# self.noise.reset()

			for t in tqdm(range(self.epLen), desc="train"):

				## Select action and observe next state, reward
				if episode > self.randomEpisodes:
					action = self.select_action(state)
				else:
					action = self.random_uniform_action()

				nextState, reward = env.step(action)
				episode_reward += reward

				# Update current lowest scatter
				current = env.RMS.item()
				if current < lowest:
					lowest = current

				## Check if terminal
				if t == self.epLen - 1:
					done = 1
				else:
					done = 0

				## Cast reward and done as tensors
				reward = tensor([[reward]]).float()
				done = tensor([[done]])

				## Store transition in memory
				self.memory.push(self.Transition(state, action, reward, nextState, done))
				state_array[array_index] = state
				action_array[array_index] = action
				reward_array[array_index] = reward
				next_state_array[array_index] = nextState
				done_array[array_index] = done
				array_index += 1
				## Preform bellman update
				if episode > self.learningBegins:
					self.optimize_model()

				## Break out of loop if terminal state
				if done == 1:
					break

				state = nextState

			## Print episode statistics to console
			print(
				f'\n#:{episode}, ' \
				f'I:{initial}, ' \
				f'Lowest:{lowest}, ' \
				f'F:{current}, '\
				f'Score:{episode_reward}, ' \
				f'Epsilon: {self.epsilon}\n')

			wandb.log({
				'epsilon':self.epsilon, 
				'lowest':lowest, 
				'score':episode_reward})

			## Save
			if episode % self.saveEvery == 0:
				path = 'dataSets/3cyl0.45-0.35/'
				torch.save(self.actor.state_dict(), path + 'actor.pt')
				torch.save(self.critic.state_dict(), path + 'critic.pt')
				torch.save(self.targetActor.state_dict(), path + 'targetActor.pt')
				torch.save(self.targetCritic.state_dict(), path + 'targetCritic.pt')

				torch.save(state_array[:array_index], path + 'states')
				torch.save(action_array[:array_index], path + 'actions')
				torch.save(reward_array[:array_index], path + 'rewards')
				torch.save(next_state_array[:array_index], path + 'nextStates')
				torch.save(done_array[:array_index], path + 'dones')

			## Reduce exploration
			if episode > self.randomEpisodes:
				self.decay_epsilon()

if __name__ == '__main__':
	## env params
	NCYL = 2
	KMAX = .45
	KMIN = .35
	NFREQ = 11

	# ddpg params
	IN_SIZE 		= 2 * NCYL + NFREQ + 2
	ACTOR_N_HIDDEN 	= 2
	ACTOR_H_SIZE 	= 128
	CRITIC_N_HIDDEN = 8
	CRITIC_H_SIZE 	= 128
	N_ACTIONS 		= 2 * NCYL
	ACTION_RANGE 	= 0.5
	ACTOR_LR 		= 1e-4
	CRITIC_LR 		= 1e-3
	CRITIC_WD 		= 1e-2 		## How agressively to reduce overfitting
	GAMMA 			= 0.90 		## How much to value future reward
	TAU 			= 0.001 	## How much to update target network every step
	EPSILON 		= 1.2		## Scale of random noise
	DECAY_TIMESTEPS = 1_000 	## How slowly to reduce epsilon
	EPS_END 		= 0.02 		## Lowest epsilon allowed
	MEM_SIZE 		= 1_000_000	## How many samples in priority queue
	MEM_ALPHA 		= 0.7 		## How much to use priority queue (0 = not at all, 1 = maximum)
	MEM_BETA 		= 0.5
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
		DECAY_TIMESTEPS, 
		EPS_END, 
		MEM_SIZE, 
		BATCH_SIZE, 
		NUM_EPISODES,
		EP_LEN)

	## Setting memory hyperparameters
	agent.memory.alpha = MEM_ALPHA
	agent.memory.beta = MEM_BETA

	wandb.init(project='tscs')
	wandb.config.nCyl = NCYL
	wandb.config.kmax = KMAX
	wandb.config.kmin = KMIN
	wandb.config.nfreq = NFREQ
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
	wandb.config.decay_timesteps = DECAY_TIMESTEPS
	wandb.config.eps_end = EPS_END
	wandb.config.mem_size = MEM_SIZE
	wandb.config.alpha = MEM_ALPHA
	wandb.config.beta = MEM_BETA
	wandb.config.batch_size = BATCH_SIZE

	## Create env and agent
	env = TSCSEnv(NCYL, KMAX, KMIN, NFREQ)

	## Run training session
	agent.learn(env)