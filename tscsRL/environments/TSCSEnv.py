import matlab
import matlab.engine
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io
import pathlib
import json
import numpy as np

class BaseTSCSEnv():
	"""docstring for BaseTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		## Matlab interface
		self.eng = matlab.engine.start_matlab()
		path = str(pathlib.Path(__file__).parent.absolute())
		self.eng.addpath(path + '/objectiveFunctions')

		## Hyperparameters
		self.nCyl = nCyl
		self.F = nFreq
		self.M = matlab.double([nCyl])
		self.kMax = matlab.double([kMax])
		self.kMin = matlab.double([kMin])
		self.nFreq = matlab.double([nFreq])

		## State variables
		self.config = None
		self.TSCS = None
		self.RMS = None
		self.counter = None
		self.gradient = None

		## Image
		self.img_dim = (600, 600)
		self.transform = transforms.Compose([
			transforms.Resize(self.img_dim),
			transforms.Grayscale(),
			transforms.ToTensor()])

		## General
		self.ep_len = 100
		self.grid_size = 5.0
		self.observation_space = 2 * nCyl + nFreq + 2
		self.stepSize = stepSize

		self.info = {
			'initial': None,
			'lowest': None,
			'final': None,
			'score': None}

	def getParams(self):
		env_params = {
			'nCyl': self.nCyl,
			'kMax': np.array(self.kMax).item(),
			'kMin': np.array(self.kMin).item(),
			'nFreq': self.F,
			'ep_len': self.ep_len,
			'grid_size': self.grid_size,
			'stepSize': self.stepSize
		}
		return env_params

	def validConfig(self, config):
		"""
		Checks if config is within bounds and does not overlap cylinders
		"""
		withinBounds = False
		overlap = False
		if (-self.grid_size < config).all() and (config < self.grid_size).all():
			withinBounds = True

			coords = config.view(self.nCyl, 2)
			for i in range(self.nCyl):
				for j in range(i, self.nCyl): # O((n-1) + (n-2) + ... + 1) Runtime complexity
				# for j in range(self.nCyl): # O(n(n-1)) Runtime complexity
					if i != j:
						x1, y1 = coords[i]
						x2, y2 = coords[j]
						d = torch.sqrt((x2-x1)**2 + (y2-y1)**2)
						if d <= 2.1:
							overlap = True
		return withinBounds and not overlap

	def getConfig(self):
		"""
		Generates a configuration which is within bounds 
		and not overlaping cylinders
		"""
		while True:
			config = torch.FloatTensor(1, 2 * self.nCyl).uniform_(-self.grid_size, self.grid_size)
			if self.validConfig(config):
				break
		return config

	def getIMG(self, config):
		"""
		Produces tensor image of configuration
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-self.grid_size - 1, xmax=self.grid_size + 1)
		ax.set_ylim(ymin=-self.grid_size - 1, ymax=self.grid_size + 1)
		ax.grid()

		coords = config.view(self.nCyl, 2)
		for cyl in range(self.nCyl):
			ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=1))

		## Convert to tensor
		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		im = Image.open(buf)

		## Apply series of transformations
		X = self.transform(im)

		buf.close()
		plt.close(fig)
		return X.unsqueeze(0)

	def getReward(self, RMS, isValid):
		"""
		Computes reward based on change in scattering 
		proporitional to how close it is to zero
		"""
		reward = -RMS.item()
		reward_grad = 0
		if not isValid:
			reward += -5.0

		if self.gradient is not None:
			gradient = self.gradient.view(int(torch.Tensor(self.M)), 2)
			for i in range(int(torch.Tensor(self.M))):
				grad_norm = torch.sqrt(gradient[i, 0] ** 2 + gradient[i, 1] ** 2)
				reward_grad += - grad_norm * RMS.item()
		reward_grad = reward_grad / int(torch.Tensor(self.M))

		reward += reward_grad

		return reward

	# def getMetric(self, config):
	# 	x = self.eng.transpose(matlab.double(*config.tolist()))
	# 	tscs = self.eng.getMetric_RigidCylinder(x, self.M, self.kMax, self.kMin, self.nFreq)
	# 	tscs = torch.tensor(tscs).T
	# 	rms = tscs.pow(2).mean().sqrt().view(1,1)
	# 	return tscs, rms

	def setMetric(self, config):
		x = self.eng.transpose(matlab.double(*config.tolist()))
		tscs = self.eng.getMetric_RigidCylinder(x, self.M, self.kMax, self.kMin, self.nFreq)
		self.TSCS = torch.tensor(tscs).T
		self.RMS = self.TSCS.pow(2).mean().sqrt().view(1,1)

	def getState(self):
		state = torch.cat([self.config, self.TSCS, self.RMS, self.counter], dim=-1).float()
		return state

	def reset(self):
		"""
		Generates starting config and calculates its tscs
		"""
		self.config = self.getConfig()
		self.counter = torch.tensor([[0.0]])
		self.setMetric(self.config)
		state = self.getState()

		## Log initial scattering at beginning of episode and reset score
		self.info['initial'] = self.RMS.item()
		self.info['lowest'] = self.info['initial']
		self.info['final'] = None
		self.info['score'] = 0
		return state

	def getNextConfig(self, config, action):
		"""
		Applys action to config
		"""
		raise NotImplementedError

	def step(self, action):
		"""
		Updates the state of the environment with action. Returns next state, reward, done.
		"""
		prevConfig = self.config.clone()
		nextConfig = self.getNextConfig(self.config.clone(), action)
		isValid = self.validConfig(nextConfig)

		if isValid:
			self.config = nextConfig
		else: ## Invalid next state, do not change state variables
			self.config = prevConfig

		self.setMetric(self.config)
		self.counter += 1/self.ep_len
		nextState = self.getState()

		reward = self.getReward(self.RMS, isValid)
		self.info['score'] += reward

		done = False
		if int(self.counter.item()) == 1:
			done = True
			
		# Update current lowest scatter
		current = self.RMS.item()
		if current < self.info['lowest']:
			self.info['lowest'] = current

		self.info['final'] = current

		return nextState, reward, done, self.info

class ContinuousTSCSEnv(BaseTSCSEnv):
	"""docstring for ContinuousTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		super(ContinuousTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)

		## Dimention of action space
		self.action_space = 2 * self.nCyl

	def getNextConfig(self, config, action):
		"""
		Applys continuous action to config
		"""
		return config + action

class DiscreteTSCSEnv(BaseTSCSEnv):
	"""docstring for DiscreteTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		super(DiscreteTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)

		## Dimention of action space
		self.action_space = 4 * self.nCyl

	def getNextConfig(self, config, action):
		"""
		Applys action to config
		"""
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

if __name__ == '__main__':
	env = BaseTSCSEnv(4, 0.45, 0.35, 11, 0.5)

	state = env.reset()

	print(state)

	env.step(2)
