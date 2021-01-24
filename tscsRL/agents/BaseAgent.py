from collections import namedtuple
from tscsRL.agents.memory import NaivePrioritizedBuffer
from tscsRL.utils import dictToJson
import torch
import os
import json
from tqdm import tqdm
import wandb
from numpy import prod
import gym
from tscsRL.utils import plot
import numpy as np

def default_params():
	params = {
		'gamma': 0.9,			## Valuation of future reward [1, 0]
		'batch_size': 64,		## Number of samples per gradient update
		'mem_size': 1_000_000,	## Maximum datapoints to store in queue
		'mem_alpha': 0.7,		## How much to use priority queue [1, 0]
		'mem_beta': 0.5,		## How agressivly to apply importance sampling
		'num_episodes':20_000,	## Number of episodes to train for
		'save_every':500,		## Save checkpoint every
		'random_episodes':0,	## Number of episodes to only select random actions
		'learning_begins':0,	## Delay learning by this many episodes
		'save_data': False,		## Save s, a, r, s_, done transitions?
		'use_wandb': False		## Log data to weights and biases logger
	}
	return params

class BaseAgent():
	"""docstring for BaseAgent"""
	def __init__(self, observation_space, action_space, params, run_name):
		super(BaseAgent, self).__init__()
		## Environment info
		self.observation_space = observation_space
		self.action_space = action_space
		
		"""
		This bit of code is ugly but nescesary for getting the dimention of
		the action and observation space for data saving. We can make this
		nicer in future versions.
		"""
		self.observation_dim = prod(observation_space.shape)

		if isinstance(self.action_space, gym.spaces.Box):
			self.action_dim = prod(self.action_space.shape)
		elif isinstance(self.action_space, gym.spaces.Discrete):
			self.action_dim = 1
		else:
			print('Unrecognized action space')

		## Hyperparameters
		self.params = params
		self.gamma = params['gamma']
		self.batch_size = params['batch_size']

		## Memory
		self.Transition = namedtuple('Transition', ('s', 'a', 'r', 's_', 'done'))
		self.memory = NaivePrioritizedBuffer(self.params['mem_size'], self.params['mem_alpha'])
		self.mem_beta = self.params['mem_beta']

		## Name of run to save
		self.run_name = run_name

	def select_action(self, state):
		raise NotImplementedError

	def random_action(self):
		raise NotImplementedError

	def extract_tensors(self, batch):
		batch = self.Transition(*zip(*batch))
		s = torch.cat(batch.s)
		a = torch.cat(batch.a)
		r = torch.cat(batch.r)
		s_ = torch.cat(batch.s_)
		done = torch.cat(batch.done)
		return s, a, r, s_, done

	def optimize_model(self):
		raise NotImplementedError

	def save_checkpoint(self, path, episode):
		raise NotImplementedError

	def load_checkpoint(self, path):
		raise NotImplementedError

	def finish_episode(self):
		raise NotImplementedError

	def getLogger(self):
		raise NotImplementedError

	def report(self, data, logger):
		raise NotImplementedError

	def learn(self, env):
		path = 'results/' + self.run_name + '/'
		checkpoint_path = path + 'checkpoints/'
		data_path = path + 'data/'

		## Make directory for run
		os.makedirs(path, exist_ok=False)
		os.makedirs(checkpoint_path, exist_ok=True)

		env_params = env.getParams()
		## Save settings for env and agent at beginning of run
		dictToJson(env_params, path + 'env_params.json')
		dictToJson(self.params, path + 'agent_params.json')

		run_params = {**env_params, **self.params}		

		## Prepare to save data from run
		if self.params['save_data']:
			os.makedirs(data_path, exist_ok=True)
			array_size = self.params['num_episodes'] * env.ep_len 
			state_array = torch.zeros(array_size, self.observation_dim)
			action_array = torch.zeros(array_size, self.action_dim)
			reward_array = torch.zeros(array_size, 1)
			next_state_array = torch.zeros(array_size, self.observation_dim)
			done_array = torch.zeros(array_size, 1)
			array_index = 0

		## Initialize logger to track episode statistics
		logger = None
		if self.params['use_wandb']:
			logger = self.getLogger(run_params, self.run_name)

		for episode in range(self.params['num_episodes']):

			## Reset environment to starting state
			state = env.reset()
			lowest = env.RMS.item()
			episode_reward = 0
			num_invalid = 0
			for t in tqdm(range(env.ep_len + 1), desc="train"):

				## Select action and observe next state, reward
				if episode >= self.params['random_episodes']:
					action = self.select_action(state)
				else:
					action = self.random_action()

				# nextState, reward, done, info = env.step(action)
				nextState, reward, done, info, inValid = env.step(action)

				num_invalid += inValid

				if env.RMS.item() < lowest:
					lowest = env.RMS.item()

				## Cast reward and done as tensors
				reward = torch.tensor([[reward]]).float()
				episode_reward += reward
				done = torch.tensor([[1 if done == True else 0]])

				## Store transition in memory
				self.memory.push(self.Transition(state, action, reward, nextState, done))
				if self.params['save_data']:
					state_array[array_index] = state
					action_array[array_index] = action
					reward_array[array_index] = reward
					next_state_array[array_index] = nextState
					done_array[array_index] = done
					array_index += 1

				## Preform update
				if episode >= self.params['learning_begins']:
					self.optimize_model()

				state = nextState
				if done.item() == 1:
					break

			## Send data to the report function which logs data and prints statistics about the algorithm and environment
			data = {'episode': episode, **info}
			self.report(data, logger)

			## Saving model checkpoint and data
			if (episode+1) % self.params['save_every'] == 0:
				self.save_checkpoint(checkpoint_path, episode)

			if self.params['save']:
				self.params['reward'].append(episode_reward)
				self.params['lowest'].append(lowest)
				self.params['invalid'].append(num_invalid)

			## Reduce exploration
			if episode >= self.params['random_episodes']:
				self.finish_episode()

		if self.params['plot_hpc']:
			plot('reward', self.params['reward'], path)
			plot('lowest', self.params['lowest'], path)
			plot('invalid', self.params['invalid'], path)
			np.savetxt(path+"reward.csv", np.array(self.params['reward']), delimiter=",")
			np.savetxt(path + "lowest.csv", np.array(self.params['lowest']), delimiter=",")
			np.savetxt(path + "invalid.csv", np.array(self.params['invalid']), delimiter=",")