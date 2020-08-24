import matlab.engine
import torch

## 4 Cylinder TSCS calculator
class TSCSEnv():
	"""docstring for TSCSEnv"""
	def __init__(self):
		## Matlab interface
		self.eng = matlab.engine.start_matlab()
		self.eng.addpath('TSCS')
		self.nCyl = 4
		self.stepSize = 0.5

		## State variables
		self.img = None
		self.config = None
		self.TSCS = None

	def validConfig(self, config):
		"""
		Checks if config is within bounds and does not overlap cylinders
		"""
		coords = config.view(self.nCyl, 2)

		overlap = False
		for i in range(self.nCyl):
			for j in range(self.nCyl):
				if i != j:
					x1, y1 = coords[i]
					x2, y2 = coords[j]
					d = torch.sqrt((x2-x1)**2 + (y2-y1)**2)
					if d <= 1:
						overlap = True
		return not overlap

	def getConfig(self):
		valid = False
		while not valid:
			config = torch.FloatTensor(1, 8).uniform_(-5, 5)
			if self.validConfig(config):
				break
		return config

	def getTSCS(self, config):
		tscs = self.eng.getTSCS4CYL(*self.config.squeeze(0).tolist())
		return torch.tensor(tscs).T

	def getReward(self, TSCS, nextTSCS):
		reward = (TSCS.sum() - nextTSCS.sum()).mean().item()
		return reward

	def reset(self):
		self.config = self.getConfig()
		self.TSCS = self.getTSCS(self.config)
		state = (self.config, self.TSCS)
		return state

	def getNextConfig(self, config, action):
		## Applys action to config
		coords = config.view(self.nCyl, 2)
		cyl = int(action/4)
		direction = action % 4
		if direction == 0:
			coords[cyl, 0] -= self.stepSize
		if direction == 1:
			coords[cyl, 1] += self.stepSize
		if direction == 2:
			coords[cyl, 0] += self.stepSize
		if direction == 3:
			coords[cyl, 1] -= self.stepSize
		nextConfig = coords.view(1, 2 * self.nCyl)
		return nextConfig

	def step(self, action):
		nextConfig = self.getNextConfig(self.config, action)

		## If the config after applying the action is not valid
		# we revert back to previous state and give 0 reward
		done = False
		if not self.validConfig(nextConfig):
			reward = torch.tensor([[-10.0]])
			done = True
		else:
			self.config = nextConfig
			nextTSCS = self.getTSCS(self.config)
			reward = self.getReward(self.TSCS, nextTSCS)
			self.TSCS = nextTSCS

		state = (self.config, self.TSCS)
		return state, reward, done

if __name__ == '__main__':
	env = TSCSEnv()
	state = env.reset()
	config, tscs = state
	print(config)
	print(tscs)

