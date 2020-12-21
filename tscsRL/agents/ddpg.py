from tscsRL.agents import BaseAgent
from tscsRL.agents.models.ActorCritic import Actor, Critic

import torch
from torch import tensor
import torch.nn.functional as F
from copy import deepcopy
import numpy as np

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
	def __init__(self, observation_space, action_space, stepSize, params, run_name):
		super(DDPGAgent, self).__init__(observation_space, action_space, stepSize, params, run_name)

		## Defining networks
		self.actor = Actor(
			observation_space, 
			self.params['actor_n_hidden'],
			self.params['actor_h_size'],
			action_space,
			stepSize,
			self.params['actor_lr'])

		self.critic = Critic(
			observation_space,
			self.params['critic_n_hidden'],
			self.params['critic_h_size'],
			action_space,
			self.params['critic_lr'],
			self.params['critic_wd'])

		## Target networks
		self.targetActor = deepcopy(self.actor)
		self.targetCritic = deepcopy(self.critic)

		## cuda:0 or cpu
		self.device = self.actor.device

		## Spaces
		self.observation_space = observation_space
		self.action_space = action_space
		self.stepSize = stepSize

		## Noise decay rate
		self.noise_scale = self.params['noise_scale']
		self.noise_decay_rate = (self.params['noise_scale'] - self.params['noise_scale_end']) / self.params['decay_timesteps']

	def finish_episode(self):
		self.noise_scale -= self.noise_decay_rate
		self.noise_scale = max(self.noise_scale, self.params['noise_scale_end'])

	def report(self, data, logger):
		episode = data['episode']
		initial = data['initial']
		lowest = data['lowest']
		final = data['final']
		score = data['score']
		data['noise_scale'] = self.noise_scale

		print(f'#:{episode}, I:{initial}, Lowest:{lowest}, F:{final}, Score:{score}, Exploration: {self.noise_scale}')
		logger.log({'epsilon':self.noise_scale, 'lowest':lowest, 'score':score})

	def select_action(self, state):
		with torch.no_grad():
			noise = np.random.normal(0, 1, size=(1, self.action_space)) * self.noise_scale
			action = self.actor(state).cpu() + noise
			action.clamp_(-self.stepSize, self.stepSize)
		return action

	def random_action(self):
		action = np.random.uniform(
			-self.stepSize, 
			self.stepSize, 
			size=(1, self.action_space))
		action = torch.tensor(action)
		return action

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