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
	def __init__(self):
		## Matlab interface
		self.eng = matlab.engine.start_matlab()
		self.eng.addpath('TSCS')
		self.nCyl = 4
		self.stepSize = 1

		## State variables
		self.config = None
		self.TSCS = None
		self.RMS = None
		self.img = None

		## Counter which tracks number of steps in a row
		# under 1 RMS
		self.counter = 0

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
						if d <= 2:
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
		# ax.axis('off')

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

	def render(self):
		"""
		Shows config in image form to the screen
		"""
		plt.imshow(self.img.view(self.img_dim, self.img_dim))
		plt.show()

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
		# OUTSIDE_RANGE_REWARD = -1500
		# ILLEGAL_MOVE_REWARD = -1000
		# DESIRED_RANGE_REWARD = 10000
		# if RMS >= 2: ## Outside acceptable range
		# 	self.counter += 1
		# 	if self.counter == 10: ## Above 2 RMS for 10 steps
		# 		reward = OUTSIDE_RANGE_REWARD
		# 		done = True
		# 	else:
		# 		reward = -1
		# 		done = False
		# elif 0.1 < RMS < 2: ## RMS in good range, reset counter
		# 	if isValid: ## Good scatter range, not an illegal move
		# 		self.counter = 0
		# 		reward = 0.2**(RMS.item()-1)-1
		# 		done = False
		# 	else: ## Good scatter range but illegal move
		# 		self.counter += 1
		# 		if self.counter == 10:
		# 			reward = ILLEGAL_MOVE_REWARD
		# 			done = True
		# 		else:
		# 			reward = 0.2**(RMS.item()-1)-1
		# 			done = False
		# elif RMS <= 0.1: ## Optimal config found
		# 	reward = DESIRED_RANGE_REWARD
		# 	done = True
			
		return reward, done

	def reset(self):
		"""
		Generates starting config and calculates its tscs
		"""
		self.config = self.getConfig()
		self.TSCS = self.getTSCS(self.config)
		self.RMS = self.getRMS(self.config)
		self.img = self.getIMG(self.config)

		self.counter = 0
 
		state = (
			self.config, 
			self.TSCS, 
			self.RMS, 
			self.img)
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
		prevConfig = self.config
		nextConfig = self.getNextConfig(self.config.clone(), action)
		isValid = self.validConfig(nextConfig)

		if isValid:
			self.config = nextConfig
		else: ## Invalid next state, do not change state variables
			self.config = prevConfig

		self.TSCS = self.getTSCS(self.config)
		self.RMS = self.getRMS(self.config)
		self.img = self.getIMG(self.config)

		reward, done = self.getReward(self.RMS, isValid)

		nextState = (self.config, self.TSCS, self.RMS, self.img)
		return nextState, reward, done

if __name__ == '__main__':
	import numpy as np
	env = TSCSEnv()
	state = env.reset()

	plt.ion()
	fig = plt.figure()
	ax = fig.add_subplot()
	myobj = ax.imshow(state[3].view(50, 50))
	print(f"RMS: {round(state[2].item(),2)}")

	for t in range(100):
		# action = np.random.randint(16)
		action = int(input("ACTION: "))
		print(f"Action: {action}")
		state, reward, done = env.step(action)
		myobj.set_data(state[3].view(50, 50))
		fig.canvas.draw()
		fig.canvas.flush_events()

		print(f"RMS: {round(state[2].item(),2)}")
		print(f"Reward: {reward}")
		if done:
			break