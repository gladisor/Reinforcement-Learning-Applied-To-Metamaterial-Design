from env import DistributedTSCSEnv
from models import Actor, Critic
import ray
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time
import copy

@ray.remote
class RolloutWorker():
	"""
	This is a class which is used to generate episodes of interaction data
	in paralell.

	Attributes:
		env (DistributedTSCSEnv): The environment which provides feedback from actions
		policy (nn.Module): The policy which maps states to actions
	"""
	def __init__(self, config):
		## Defining environment for this RolloutWorker
		self.env = DistributedTSCSEnv(config['env_config'])

		## Defining policy for this RolloutWorker
		self.policy = Actor(
			self.env.observation_space.shape[1],
			self.env.action_space.shape[1],
			config['model']['actor_fc'],
			self.env.action_space.high.max())

	def compute_action(self, state, sigma):
		"""
		Computes the action to take in the current state with noise scale of sigma.

		Parameters:
			state (torch.tensor): Vector containing information about the state of the environment
			sigma (float): standard deviation of noise to add to action

		Returns:
			action (numpy.array): Action to pass to environment
		"""
		with torch.no_grad():
			noise = np.random.normal(0, sigma, self.env.action_space.shape)
			action = self.policy(torch.tensor(state).float()).numpy() + noise
		return action

	def rollout_episode(self, sigma):
		"""
		Generates data for the execution of one episode.

		Parameters:
			sigma (float): standard deviation of noise to add to action

		Returns:
			data (dict): dictionary containing numpy arrays holding data for each field.
		"""
		data = {
			'states': np.ndarray(shape=(self.env.episodeLength, self.env.observation_space.shape[1])),
			'actions': np.ndarray(shape=(self.env.episodeLength, self.env.action_space.shape[1])),
			'rewards': np.ndarray(shape=(self.env.episodeLength, 1)),
			'next_states': np.ndarray(shape=(self.env.episodeLength, self.env.observation_space.shape[1])),
			'dones': np.ndarray(shape=(self.env.episodeLength, 1))}

		state = self.env.reset()
		done = False
		idx = 0
		while not done:
			action = self.compute_action(state, sigma)
			next_state, reward, done, info = self.env.step(action)

			data['states'][idx] = state
			data['actions'][idx] = action
			data['rewards'][idx] = reward
			data['next_states'][idx] = next_state
			data['dones'][idx] = done
			idx += 1

			state = next_state
		return data

class Learner():
	"""
	This is the centralized learner for DistributedDDPG.
	"""
	def __init__(self, config):
		self.env = DistributedDDPG(config['env_config'])

		self.actor = Actor(
			self.env.observation_space.shape[1],
			self.env.action_space.shape[1],
			config['model']['actor_fc'],
			self.env.action_space.high.max()).cuda()

		self.critic = Critic(
			env.observation_space.shape[1],
			env.action_space.shape[1],
			config['model']['critic_fc']).cuda()

		self.targetActor = copy.deepcopy(self.actor)
		self.targetCritic = copy.deepcopy(self.critic)

		self.actorOpt = Adam(
			self.actor.parameters(), 
			lr=config['model']['actor_lr'])

		self.criticOpt = Adam(
			self.critic.parameters(),
			lr=config['model']['critic_lr'],
			weight_decay=config['model']['critic_wd'])

		self.num_epochs = config['num_epochs']
		self.batch_size = config['batch_size']
		self.num_learner_workers = config['num_learner_workers']
		self.gamma = config['gamma']
		self.tau = config['tau']

	def soft_update(self, target, source):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

	def optimize_model(self, data):
		episode_data = {
			'states': np.concatenate([data[i]['states'] for i in range(len(data))]),
			'actions': np.concatenate([data[i]['actions'] for i in range(len(data))]),
			'rewards': np.concatenate([data[i]['rewards'] for i in range(len(data))]),
			'next_states': np.concatenate([data[i]['next_states'] for i in range(len(data))]),
			'dones': np.concatenate([data[i]['dones'] for i in range(len(data))])}

		train = Episodes(episode_data)
		trainLoader = DataLoader(
			train, 
			batch_size=self.batch_size,
			num_workers=self.num_learner_workers,
			shuffle=True)

		for epoch in range(self.num_epochs):
			for s, a, r, s_, done in trainLoader:
				s, a, r, s_, done = s.cuda(), a.cuda(), r.cuda(), s_.cuda(), done.cuda()

				## Compute target
				maxQ = self.targetCritic(s_, self.targetActor(s_).detach())
				target_q = r + (1.0 - done) * self.gamma * maxQ

				## Update the critic network
				self.criticOpt.zero_grad()
				current_q = self.critic(s, a)
				criticLoss = F.smooth_l1_loss(current_q, target_q.detach())
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

class Episodes(Dataset):
	def __init__(self, data):
		shuffled_idx = torch.randperm(len(data['states']))
		for key in data.keys():
			data[key] = torch.tensor(data[key]).float()
			data[key] = data[key][shuffled_idx]

		self.data = data

	def __getitem__(self, idx):
		return [self.data[key][idx] for key in self.data.keys()]

	def __len__(self):
		return len(self.data['states'])

if __name__ == '__main__':
	config = {
		'env_config': {
			'nCyl': 4,
			'k0amax': 0.45,
			'k0amin': 0.35,
			'nFreq': 11,
			'actionRange': 0.2,
			'episodeLength': 100},
		'model': {
			'actor_fc': [128] * 2,
			'actor_lr': 1e-4,
			'critic_fc': [128] * 4,
			'critic_lr': 1e-3,
			'critic_wd': 1e-2
		},
		'num_epochs': 10,
		'batch_size': 64,
		'num_learner_workers': 2,
		'gamma': 0.90,
		'tau': 0.001
	}

	ray.init()

	## Data gathering
	agents = [RolloutWorker.remote(config) for _ in range(8)]
	data = []
	for _ in range(5):
		futures = [agent.rollout_episode.remote(1.0) for agent in agents]
		start = time.time()
		data += ray.get(futures)
		print(f'Parallel env data generation: {time.time() - start}')

	## Data processing
	start = time.time()

	episode_data = {
		'states': np.concatenate([data[i]['states'] for i in range(len(data))]),
		'actions': np.concatenate([data[i]['actions'] for i in range(len(data))]),
		'rewards': np.concatenate([data[i]['rewards'] for i in range(len(data))]),
		'next_states': np.concatenate([data[i]['next_states'] for i in range(len(data))]),
		'dones': np.concatenate([data[i]['dones'] for i in range(len(data))])}

	print(f'Data processing time: {time.time() - start}')

	## Data loading
	start = time.time()

	train = Episodes(episode_data)
	trainLoader = DataLoader(train, batch_size=64, num_workers=2)
	print(f'Loading data took: {time.time() - start} seconds')

	ray.shutdown()
