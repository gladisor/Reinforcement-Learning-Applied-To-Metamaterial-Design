import matlab.engine
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torchvision import transforms
from PIL import Image
import io

## 4 Cylinder TSCS
class TSCSEnv():
	"""docstring for TSCSEnv"""
	def __init__(self, nCyl=4, stepSize=0.5):
		## Matlab interface
		self.eng = matlab.engine.start_matlab()
		self.eng.addpath('TSCS')

		## Hyperparameters
		self.nCyl = nCyl
		self.stepSize = stepSize

		## State variables
		self.config = None
		self.TSCS = None
		self.RMS = None
		self.img = None
		self.counter = None

		## Image transform
		self.img_dim = 50
		self.transform = transforms.Compose([
			transforms.Resize((self.img_dim, self.img_dim)),
			transforms.Grayscale(),
			transforms.ToTensor()])

	def validConfig(self, config):
		"""
		Checks if config is within bounds and does not overlap cylinders
		"""
		withinBounds = False
		overlap = False
		if (-5 < config).all() and (config < 5).all():
			withinBounds = True

			coords = config.view(self.nCyl, 2)
			for i in range(self.nCyl):
				for j in range(self.nCyl):
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
			config = torch.FloatTensor(1, 8).uniform_(-5, 5)
			if self.validConfig(config):
				break
		return config

	def getTSCS(self, config):
		## Gets tscs of configuration from matlab
		tscs = self.eng.getTSCS4CYL(*config.squeeze(0).tolist())
		return torch.tensor(tscs).T

	def getRMS(self, config):
		## Gets rms of configuration from matlab
		rms = self.eng.getRMS4CYL(*config.squeeze(0).tolist())
		return torch.tensor([[rms]])

	def getIMG(self, config):
		"""
		Produces tensor image of configuration
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-6, xmax=6)
		ax.set_ylim(ymin=-6, ymax=6)
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

	def getTime(self):
		return self.counter/100

	def getReward(self, RMS, isValid):
		"""
		Computes reward based on change in scattering 
		proporitional to how close it is to zero
		"""
		if isValid:
			reward = 0.2**(RMS.item()-1)-1
		else:
			reward = -1
			
		done = False
		return reward, done

	def reset(self):
		"""
		Generates starting config and calculates its tscs
		"""
		self.config = self.getConfig()
		self.TSCS = self.getTSCS(self.config)
		self.RMS = self.getRMS(self.config)
		# self.img = self.getIMG(self.config)
		self.counter = torch.tensor([[0.0]])
		time = self.getTime()

		state = (
			self.config, 
			self.TSCS, 
			self.RMS,
			# self.img,
			time)
		return state

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

		self.TSCS = self.getTSCS(self.config)
		self.RMS = self.getRMS(self.config)
		# self.img = self.getIMG(self.config)
		self.counter += 1
		time = self.getTime()

		reward, done = self.getReward(self.RMS, isValid)

		nextState = (
			self.config,
			self.TSCS,
			self.RMS,
			# self.img,
			time)
		return nextState, reward, done

if __name__ == '__main__':
	import numpy as np
	env = TSCSEnv()
	state = env.reset()

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot()

	img = env.getIMG(state[0])
	myobj = ax.imshow(img.view(env.img_dim, env.img_dim))
	print(f"RMS: {round(state[2].item(), 2)}")

	for t in range(100):
		action = np.random.randint(16)
		print(f"Action: {action}")
		state, reward, done = env.step(action)

		img = env.getIMG(state[0])
		myobj.set_data(img.view(env.img_dim, env.img_dim))
		fig.canvas.draw()
		fig.canvas.flush_events()
		plt.pause(0.05)

		print(f"RMS: {round(state[2].item(), 2)}")
		print(f"Reward: {reward}")
		if done:
			break