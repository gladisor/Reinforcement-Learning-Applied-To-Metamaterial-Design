from tscsRL.agents import BaseAgent
from tscsRL.agents.models.ActorCritic import Actor, Critic, ImageActor, ImageCritic

import torch
from torch import tensor
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import wandb
from collections import namedtuple
import os
from tscsRL.utils import dictToJson
from tqdm import tqdm
from numpy import prod
from tscsRL.utils import plot


def default_params():
	params = {
		'actor_n_hidden': 2,		## Layers in actor network
		'actor_h_size': 128,		## Neurons in actor network
		'actor_lr': 1e-4,			## Learning rate of actor network
		'critic_n_hidden': 8,		## Layers in critic network
		'critic_h_size': 128,		## Neurons in critic network
		'critic_lr':1e-3,			## Learning rate of critic network
		'critic_wd':1e-2,			## Weight decay of critic network
		'tau': 0.001,				## Rate at which the target networks track the base networks
		'noise_scale': 1.2,			## Scale of N(0, 1) noise to apply to action
		'decay_timesteps': 8000,	## Rate at which the noise scale drops to the minimum
		'noise_scale_end': 0.02		## Minimum noise scale rate
		}

	base_params = BaseAgent.default_params()
	params.update(base_params)
	return params

class DDPGAgent(BaseAgent.BaseAgent):
	"""docstring for DDPGAgent"""
	def __init__(self, observation_space, action_space, params, run_name):
		super(DDPGAgent, self).__init__(observation_space, action_space, params, run_name)
		## Spaces
		self.action_high = torch.tensor(self.action_space.high)
		self.action_low = torch.tensor(self.action_space.low)
		## Note: Does not support action range which is not symmetrical around 0
		self.action_range = (self.action_high - self.action_low)/2

		## Hyperparameters
		self.tau = self.params['tau']

		## Defining networks
		self.actor = Actor(
			self.observation_dim, 
			self.params['actor_n_hidden'],
			self.params['actor_h_size'],
			self.action_dim,
			self.action_range,
			self.params['actor_lr'])

		self.critic = Critic(
			self.observation_dim,
			self.params['critic_n_hidden'],
			self.params['critic_h_size'],
			self.action_dim,
			self.params['critic_lr'],
			self.params['critic_wd'])

		## Target networks
		self.targetActor = deepcopy(self.actor)
		self.targetCritic = deepcopy(self.critic)

		## cuda:0 or cpu
		self.device = self.actor.device

		## Noise
		self.noise_scale = self.params['noise_scale']
		self.noise_scale_end = self.params['noise_scale_end']
		self.noise_decay_rate = (self.noise_scale - self.noise_scale_end) / self.params['decay_timesteps']

	def finish_episode(self):
		self.noise_scale -= self.noise_decay_rate
		self.noise_scale = max(self.noise_scale, self.noise_scale_end)

	def getLogger(self, config, name):
		## Specify project name for wandb to store runs from this algorithm in
		return wandb.init(project='tscs', config=config, name=name)

	def report(self, data, logger):
		data['noise_scale'] = self.noise_scale
		print(data)
		if self.params['use_wandb']:
			logger.log(data)

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, 1, size=self.action_space.shape) * self.noise_scale
			action = self.actor(state).cpu() + noise
			action = torch.max(torch.min(action.double(), self.action_high.double()), self.action_low.double())
		return action

	def random_action(self):
		return torch.tensor(self.action_space.sample())

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	def optimize_model(self):
		if self.memory.can_provide_sample(self.batch_size):
			## Get data from memory
			batch, indices, weights = self.memory.sample(self.batch_size, self.mem_beta)
			s, a, r, s_, done = self.extract_tensors(batch)
			r = r.to(self.device)
			done = done.to(self.device)
			weights = tensor([weights]).to(self.device)

			## Compute target
			maxQ = self.targetCritic(s_, self.targetActor(s_).detach())
			target_q = r + (1.0 - done) * self.gamma * maxQ

			## Update the critic network
			self.critic.opt.zero_grad()
			current_q = self.critic(s, a)
			criticLoss = weights @ F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')
			criticLoss.backward()
			self.critic.opt.step()

			## Update the actor network
			self.actor.opt.zero_grad()
			actorLoss = -self.critic(s, self.actor(s)).mean()
			actorLoss.backward()
			self.actor.opt.step()

			## Copy policy weights over to target net
			self.soft_update(self.targetActor, self.actor)
			self.soft_update(self.targetCritic, self.critic)

			## Updating priority of transition by last absolute td error
			td = torch.abs(target_q - current_q).detach()
			self.memory.update_priorities(indices, td + 1e-5)
	
	def save_checkpoint(self, path, episode):
		torch.save(self.actor.state_dict(), path + f'actor{episode}.pt')
		torch.save(self.critic.state_dict(), path + f'critic{episode}.pt')
		torch.save(self.targetActor.state_dict(), path + f'targetActor{episode}.pt')
		torch.save(self.targetCritic.state_dict(), path + f'targetCritic{episode}.pt')

	def load_checkpoint(self, path, episode):
		self.actor.load_state_dict(torch.load(path + f'actor{episode}.pt'))
		self.critic.load_state_dict(torch.load(path + f'critic{episode}.pt'))
		self.targetActor.load_state_dict(torch.load(path + f'targetActor{episode}.pt'))
		self.targetCritic.load_state_dict(torch.load(path + f'targetCritic{episode}.pt'))


class ImageDDPGAgent(DDPGAgent):
	"""docstring for DDPGAgent"""
	def __init__(self, observation_space, action_space, params, run_name):
		super(ImageDDPGAgent, self).__init__(observation_space, action_space, params, run_name)

		self.Transition = namedtuple('Transition', ('s', 'img', 'a', 'r', 's_', 'img_', 'done'))
		## Defining networks
		self.actor = ImageActor(
			self.observation_dim,
			self.params['actor_n_hidden'],
			self.params['actor_h_size'],
			self.action_dim,
			self.action_range,
			self.params['actor_lr'])

		self.critic = ImageCritic(
			self.observation_dim,
			self.params['critic_n_hidden'],
			self.params['critic_h_size'],
			self.action_dim,
			self.params['critic_lr'],
			self.params['critic_wd'])

		## Target networks
		self.targetActor = deepcopy(self.actor)
		self.targetCritic = deepcopy(self.critic)

	def select_action(self, img, state):
		with torch.no_grad():
			noise = np.random.normal(0, 1, size=self.action_space.shape) * self.noise_scale
			action = self.actor(img, state).cpu() + noise
			action = torch.max(torch.min(action.double(), self.action_high.double()), self.action_low.double())
		return action

	def optimize_model(self):
		if self.memory.can_provide_sample(self.batch_size):
			## Get data from memory
			batch, indices, weights = self.memory.sample(self.batch_size, self.mem_beta)
			s, img, a, r, s_, img_, done = self.extract_tensors(batch)
			r = r.to(self.device)
			done = done.to(self.device)
			weights = tensor([weights]).to(self.device)

			## Compute target
			maxQ = self.targetCritic(img_, s_, self.targetActor(img_, s_).detach())
			target_q = r + (1.0 - done) * self.gamma * maxQ

			## Update the critic network
			self.critic.opt.zero_grad()
			current_q = self.critic(img, s, a)
			criticLoss = weights @ F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')
			criticLoss.backward()
			self.critic.opt.step()

			## Update the actor network
			self.actor.opt.zero_grad()
			actorLoss = -self.critic(img, s, self.actor(img, s)).mean()
			actorLoss.backward()
			self.actor.opt.step()

			## Copy policy weights over to target net
			self.soft_update(self.targetActor, self.actor)
			self.soft_update(self.targetCritic, self.critic)

			## Updating priority of transition by last absolute td error
			td = torch.abs(target_q - current_q).detach()
			self.memory.update_priorities(indices, td + 1e-5)

	def extract_tensors(self, batch):
		batch = self.Transition(*zip(*batch))
		s = torch.cat(batch.s)
		img = torch.cat(batch.img)
		a = torch.cat(batch.a)
		r = torch.cat(batch.r)
		s_ = torch.cat(batch.s_)
		img_ = torch.cat(batch.img_)
		done = torch.cat(batch.done)
		return s, img, a, r, s_, img_, done

	def learn(self, env):
		path = 'results/' + self.run_name + '/'
		checkpoint_path = path + 'checkpoints/'

		## Make directory for run
		os.makedirs(path, exist_ok=False)
		os.makedirs(checkpoint_path, exist_ok=True)

		env_params = env.getParams()
		## Save settings for env and agent at beginning of run
		dictToJson(env_params, path + 'env_params.json')
		dictToJson(self.params, path + 'agent_params.json')

		run_params = {**env_params, **self.params}

		## Initialize logger to track episode statistics
		logger = None
		if self.params['use_wandb']:
			logger = self.getLogger(run_params, self.run_name)

		for episode in range(self.params['num_episodes']):

			## Reset environment to starting state
			state = env.reset()
			img = env.img
			lowest = env.RMS.item()
			episode_reward = 0
			for t in tqdm(range(env.ep_len + 1), desc="train"):

				## Select action and observe next state, reward
				if episode >= self.params['random_episodes']:
					action = self.select_action(img, state)
				else:
					action = self.random_action()

				nextState, reward, done, info = env.step(action)
				nextImg = env.img

				if env.RMS.item() < lowest:
					lowest = env.RMS.item()

				## Cast reward and done as tensors
				reward = torch.tensor([[reward]]).float()
				episode_reward += reward
				done = torch.tensor([[1 if done == True else 0]])

				## Store transition in memory
				self.memory.push(self.Transition(state, img, action, reward, nextState, nextImg, done))

				## Preform update
				if episode >= self.params['learning_begins']:
					self.optimize_model()

				state = nextState
				img = nextImg

				if done.item() == 1:
					break

			## Send data to the report function which logs data and prints statistics about the algorithm and environment
			data = {'episode': episode, **info}
			self.report(data, logger)


			## Saving model checkpoint and data
			if (episode+1) % (self.params['save_every']) == 0:
				self.save_checkpoint(checkpoint_path, episode)

			if self.params['save']:
				self.params['reward'].append(episode_reward)
				self.params['lowest'].append(lowest)

			## Reduce exploration
			if episode >= self.params['random_episodes']:
				self.finish_episode()

		if self.params['plot_hpc']:
			plot('reward', self.params['reward'], path)
			plot('lowest', self.params['lowest'], path)
