from tscsRL.agents import BaseAgent
from tscsRL.agents.models.ActorCritic import Actor, Critic

import torch
from torch import tensor
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import wandb

def default_params():
	params = {
		'actor_n_hidden': 2,
		'actor_h_size': 128,
		'actor_lr': 1e-4,
		'critic_n_hidden': 8,
		'critic_h_size': 128,
		'critic_lr':1e-3,
		'critic_wd':1e-2,
		'gamma': 0.90,
		'tau': 0.001,
		'noise_scale': 1.2,
		'decay_timesteps': 8000,
		'noise_scale_end': 0.02,
		'batch_size': 64}

	base_params = BaseAgent.default_params()
	params.update(base_params)
	return params

class DDPGAgent(BaseAgent.BaseAgent):
	"""docstring for DDPGAgent"""
	def __init__(self, observation_space, action_space, params, run_name):
		super(DDPGAgent, self).__init__(observation_space, action_space, params, run_name)
		## Spaces
		# self.action_range = self.params['action_range']
		self.action_high = torch.tensor(self.action_space.high)
		self.action_low = torch.tensor(self.action_space.low)
		action_range = (self.action_high - self.action_low)/2

		## Defining networks
		self.actor = Actor(
			self.observation_dim, 
			self.params['actor_n_hidden'],
			self.params['actor_h_size'],
			self.action_dim,
			action_range,
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

		## Noise decay rate
		self.noise_scale = self.params['noise_scale']
		self.noise_decay_rate = (self.params['noise_scale'] - self.params['noise_scale_end']) / self.params['decay_timesteps']

	def finish_episode(self):
		self.noise_scale -= self.noise_decay_rate
		self.noise_scale = max(self.noise_scale, self.params['noise_scale_end'])

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
			# action.clamp_(self.action_space.low, self.action_space.high)
			action = torch.max(torch.min(action, self.action_high), self.action_low)
		return action

	def random_action(self):
		return torch.tensor(self.action_space.sample())

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.params['tau']) + param.data * self.params['tau'])

	def optimize_model(self):
		if self.memory.can_provide_sample(self.params['batch_size']):
			## Get data from memory
			batch, indices, weights = self.memory.sample(self.params['batch_size'], self.mem_beta)
			s, a, r, s_, done = self.extract_tensors(batch)
			weights = tensor([weights]).to(self.device)

			## Compute target
			maxQ = self.targetCritic(s_, self.targetActor(s_).detach())
			target_q = r.to(self.device) + (1.0 - done.to(self.device)) * self.params['gamma'] * maxQ

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