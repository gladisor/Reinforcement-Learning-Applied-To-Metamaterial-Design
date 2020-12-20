from env import TSCSEnv
from ddpg import DDPG
import wandb
import matlab
import torch

class contGradientEnv(TSCSEnv):
	def __init__(self, nCyl, k0amax, k0amin, nFreq):
		super(contGradientEnv, self).__init__(nCyl, k0amax, k0amin, nFreq)
		self.action_space = nCyl * 2
		self.observation_space = nCyl * 4 + nFreq + 2

	def getMetric(self, config):
		x = self.eng.transpose(matlab.double(*config.tolist()))
		tscs, grad = self.eng.getMetric(x, self.M, self.k0amax, self.k0amin, self.nfreq, nargout=2)
		tscs = torch.tensor(tscs).T
		rms = tscs.pow(2).mean().sqrt().view(1,1)
		grad = torch.tensor(grad).T
		return tscs, rms, grad

	def reset(self):
		"""
		Generates starting config and calculates its tscs
		"""
		self.config = self.getConfig()
		self.TSCS, self.RMS, grad = self.getMetric(self.config)
		self.counter = torch.tensor([[0.0]])
		state = torch.cat([self.config, self.TSCS, self.RMS, grad, self.counter], dim=-1).float() 
		return state

	def step(self, action):
		"""
		If the config after applying the action is not valid
		we revert back to previous state and give negative reward
		otherwise, reward is calculated by the change in scattering
		"""
		prevConfig = self.config.clone()
		nextConfig = self.getNextConfig(self.config.clone(), action)
		isValid = self.validConfig(nextConfig)

		if isValid:
			self.config = nextConfig
		else: ## Invalid next state, do not change state variables
			self.config = prevConfig

		self.TSCS, self.RMS, grad = self.getMetric(self.config)
		self.counter += 1/100

		reward = self.getReward(self.RMS, isValid)

		done = False
		if int(self.counter.item()) == 1:
			done = True
			
		nextState = torch.cat([self.config, self.TSCS, self.RMS, grad, self.counter], dim=-1).float()
		return nextState, reward

if __name__ == '__main__':
	## env params
	NCYL = 10
	KMAX = .45
	KMIN = .35
	NFREQ = 11

	## Create env and agent
	env = contGradientEnv(NCYL, KMAX, KMIN, NFREQ)
	print(env.observation_space)
	print(env.action_space)

	# ddpg params
	IN_SIZE 		= env.observation_space
	ACTOR_N_HIDDEN 	= 2
	ACTOR_H_SIZE 	= 128
	CRITIC_N_HIDDEN = 8
	CRITIC_H_SIZE 	= 128
	N_ACTIONS 		= env.action_space
	ACTION_RANGE 	= 0.5
	ACTOR_LR 		= 1e-4
	CRITIC_LR 		= 1e-3
	CRITIC_WD 		= 1e-2 		## How agressively to reduce overfitting
	GAMMA 			= 0.90 		## How much to value future reward
	TAU 			= 0.001 	## How much to update target network every step
	EPSILON 		= 1.2		## Scale of random noise
	DECAY_TIMESTEPS = 3000	 	## How slowly to reduce epsilon
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

	agent.saveEvery = 500
	agent.randomEpisodes = 0
	agent.learningBegins = 0

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

	## Run training session
	agent.learn(env)