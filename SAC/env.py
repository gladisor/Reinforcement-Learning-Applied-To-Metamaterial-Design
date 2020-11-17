import gym
import numpy as np
import matlab.engine

class TSCSEnv(gym.Env):
	"""docstring for TSCSEnv"""
	def __init__(self, params):
		## Initialize matlab
		self.eng = matlab.engine.start_matlab()
		self.eng.addpath('objectiveFunction')

		## Environment hyperparmeters
		## Python
		self.nCyl = params['nCyl']
		self.nFreq = params['nFreq']
		self.actionRange = params['actionRange']
		self.episodeLength = params['episodeLength']
		## Matlab
		self.a = matlab.double([1])
		self.aa = matlab.double([1])
		self.ha = matlab.double([0.1])
		self.M = matlab.double([self.nCyl])
		self.F = matlab.double([self.nFreq])
		self.k0amax = matlab.double([params['k0amax']])
		self.k0amin = matlab.double([params['k0amin']])
		self.c_p = matlab.double([5480])
		self.rho_sh = matlab.double([8850])

		## State variables
		self.config = None
		self.TSCS = None
		self.RMS = None
		self.timestep = None

		## Defining action and observation spaces
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, self.nCyl * 2 + self.nFreq))

		self.action_space = gym.spaces.Box(
			low=-self.actionRange,
			high=self.actionRange,
			shape=(self.nCyl * 2,))

	def validConfig(self, config):
		"""
		Checks if config is within bounds and does not overlap cylinders
		"""
		withinBounds = False
		overlap = False
		if (-5 < config).all() and (config < 5).all():
			withinBounds = True

			coords = config.reshape(self.nCyl, 2)
			for i in range(self.nCyl):
				for j in range(self.nCyl):
					if i != j:
						x1, y1 = coords[i]
						x2, y2 = coords[j]
						d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
						if d <= 2.1:
							overlap = True
		return withinBounds and not overlap

	def getConfig(self):
		"""
		Generates a configuration which is within bounds 
		and not overlaping cylinders
		"""
		while True:
			config = np.random.uniform(-5., 5., (1, self.nCyl * 2))
			if self.validConfig(config):
				break
		return config

	def getMetric(self, config):
		"""
		"""
		x = self.eng.transpose(matlab.double(*config.tolist()))

		Q_RMS, qV, kav, Q = self.eng.objectiveFunctionTSCS_RMSka_min_max(
			x,
			self.a,
			self.aa,
			self.M,
			self.ha,
			self.c_p,
			self.rho_sh,
			self.k0amax,
			self.k0amin,
			self.F,
			nargout=4)

		return np.array(Q).T, np.array([[Q_RMS]])

	def reset(self):
		"""
		"""
		self.config = self.getConfig()
		self.TSCS, self.RMS = self.getMetric(self.config)

		self.timestep = 0

		state = np.concatenate((self.config, self.TSCS), axis=-1)
		return state

	def getReward(self, TSCS, valid):
		"""
		"""
		# reward = -TSCS.mean()
		# if not valid:
		# 	reward -= 1.0

		# return reward
		meanTSCS = np.mean(TSCS, keepdims=True)
		if meanTSCS < 1.0:
			reward = -np.sqrt(meanTSCS).item()
		else:
			reward = (-0.4 * meanTSCS - 0.6).item()

		if not valid:
			reward -= 1.0
		return reward

	def step(self, action):
		"""
		"""
		nextConfig = self.config.copy() + action

		valid = False
		if self.validConfig(nextConfig):
			self.config = nextConfig
			valid = True

		self.TSCS, self.RMS = self.getMetric(self.config)
		reward = self.getReward(self.TSCS, valid)

		self.timestep += 1/self.episodeLength

		done = False
		if int(self.timestep) == 1:
			done = True

		state = np.concatenate((self.config, self.TSCS), axis=-1)
		return state, reward, done

if __name__ == '__main__':
	params = {
		'nCyl': 4,
		'nFreq': 11,
		'actionRange': 0.5,
		'episodeLength': 100,
		'k0amax': 0.45,
		'k0amin': 0.35}

	env = TSCSEnv(params)

	import time
	import matplotlib.pyplot as plt
	from utils import getIMG

	start = time.time()
	state = env.reset()

	# plt.ion()
	# fig = plt.figure()
	# ax = fig.add_subplot()
	# img = getIMG(env.config, env.nCyl)
	# myobj = ax.imshow(img.view(50, 50))

	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done = env.step(action)

		# img = getIMG(env.config, env.nCyl)
		# myobj.set_data(img.view(50, 50))
		# fig.canvas.draw()
		# fig.canvas.flush_events()
		# plt.pause(0.01)

	print(time.time() - start)