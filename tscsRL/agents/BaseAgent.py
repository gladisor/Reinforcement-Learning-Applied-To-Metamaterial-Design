from collections import namedtuple
from tscsRL.agents.memory import NaivePrioritizedBuffer
import torch
from torch import tensor
import os
import json
from tqdm import tqdm
import wandb

class BaseAgent():
	"""docstring for BaseAgent"""
	def __init__(self, observation_space, action_space, stepSize, params, run_name):
		super(BaseAgent, self).__init__()
		## Environment info
		self.observation_space = observation_space
		self.action_space = action_space
		self.stepSize = stepSize

		## Hyperparameters
		self.params = params

		## Memory
		self.Transition = namedtuple('Transition', ('s', 'a', 'r', 's_', 'done'))
		self.memory = NaivePrioritizedBuffer(self.params['mem_size'], self.params['mem_alpha'])
		self.mem_beta = self.params['mem_beta']

		## Exploration rate
		self.epsilon = self.params['epsilon']

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

	def decay_epsilon(self):
		# self.epsilon *= self.epsDecay
		self.epsilon -= (self.params['epsilon'] - self.params['eps_end']) / self.params['decay_timesteps']
		self.epsilon = max(self.epsilon, self.params['eps_end'])

	def learn(self, env):
		path = 'results/' + self.run_name + '/'
		checkpoint_path = path + 'checkpoints/'
		data_path = path + 'data/'

		## Make directory for run
		os.makedirs(path, exist_ok=False)
		os.makedirs(checkpoint_path, exist_ok=True)

		env.save_params(path)

		with open(path + 'agent_params.json', 'w') as f:
			json.dump(self.params, f)

		if self.params['save_data']:
			os.makedirs(data_path, exist_ok=True)
			array_size = self.params['num_episodes'] * env.ep_len 
			state_array = torch.zeros(array_size, self.observation_space)
			action_array = torch.zeros(array_size, self.action_space)
			reward_array = torch.zeros(array_size, 1)
			next_state_array = torch.zeros(array_size, self.observation_space)
			done_array = torch.zeros(array_size, 1)
			array_index = 0

		for episode in range(self.params['num_episodes']):

			## Reset environment to starting state
			state = env.reset()
			episode_reward = 0

			## Log initial scattering at beginning of episode
			initial = env.RMS.item()
			lowest = initial

			for t in tqdm(range(env.ep_len + 1), desc="train"):

				## Select action and observe next state, reward
				if episode > self.params['random_episodes']:
					action = self.select_action(state)
				else:
					action = self.random_action()

				nextState, reward, done = env.step(action)
				episode_reward += reward

				# Update current lowest scatter
				current = env.RMS.item()
				if current < lowest:
					lowest = current

				if done:
					done = 1
				else:
					done = 0

				## Cast reward and done as tensors
				reward = tensor([[reward]]).float()
				done = tensor([[done]])

				## Store transition in memory
				self.memory.push(self.Transition(state, action, reward, nextState, done))
				if self.params['save_data']:
					state_array[array_index] = state
					action_array[array_index] = action
					reward_array[array_index] = reward
					next_state_array[array_index] = nextState
					done_array[array_index] = done
					array_index += 1

				## Preform bellman update
				if episode > self.params['learning_begins']:
					self.optimize_model()

				state = nextState

			## Print episode statistics to console
			print(f'\n#:{episode}, I:{initial}, Lowest:{lowest}, F:{current}, Score:{episode_reward}, Epsilon: {self.epsilon}')

			wandb.log({'epsilon':self.epsilon, 'lowest':lowest, 'score':episode_reward})

			## Save
			if episode % self.params['save_every'] == 0:
				self.save_checkpoint(checkpoint_path, episode)

				if self.params['save_data']:
					torch.save(state_array[:array_index], data_path + 'states')
					torch.save(action_array[:array_index], data_path + 'actions')
					torch.save(reward_array[:array_index], data_path + 'rewards')
					torch.save(next_state_array[:array_index], data_path + 'nextStates')
					torch.save(done_array[:array_index], data_path + 'dones')

			## Reduce exploration
			if episode > self.params['random_episodes']:
				self.decay_epsilon()