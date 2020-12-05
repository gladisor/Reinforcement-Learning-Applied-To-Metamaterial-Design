from env import TSCSEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io
import torch

class RadiusEnv(TSCSEnv):
	"""docstring for RadiusEnv"""
	def __init__(self, k0amax, k0amin, nFreq, config):
		super(RadiusEnv, self).__init__(int(len(config)/2), k0amax, k0amin, nFreq)
		self.config = torch.tensor([config])

		self.minRadii = 0.1
		self.maxRadii = 2.0
		self.minDistanceBetween = 0.1

	def validRadii(self, radii):
		withinBounds = False
		overlap = False

		if (self.minRadii <= radii).all() and (radii <= self.maxRadii).all():
			withinBounds = True

			coords = self.config.reshape(self.nCyl, 2)
			for i in range(self.nCyl):
				for j in range(self.nCyl):
					if i != j:
						x1, y1 = coords[i]
						r1 = radii[0, i]
						x2, y2 = coords[j]
						r2 = radii[0, j]
						d = torch.sqrt((x2-x1)**2 + (y2-y1)**2)

						if r1 + r2 + self.minDistanceBetween >= d:
							overlap = True
		return withinBounds and not overlap

	def getInitialRadii(self):
		while True:
			radii = torch.FloatTensor(1, self.nCyl).uniform_(self.minRadii, self.maxRadii)
			if self.validRadii(radii):
				break
		return radii

	def getIMG(self, radii):
		"""
		Produces tensor image of configuration
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-6, xmax=6)
		ax.set_ylim(ymin=-6, ymax=6)
		ax.grid()

		coords = self.config.view(self.nCyl, 2)
		for cyl in range(self.nCyl):
			ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=radii[0, cyl]))

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

	def getMetric(self, radii):
		return

	def reset(self):
		self.radii = self.getInitialRadii()
		# self.TSCS, self.RMS = self.getMetric(self.radii)
		self.counter = torch.tensor([[0.0]])
		# state = torch.cat([self.radii, self.TSCS, self.RMS, self.counter], dim=-1)
		# return state

	def getNextState(self, radii, action):
		return radii + action

	def step(self, action):
		prevRadii = self.radii.clone()
		nextRadii = self.getNextState(self.radii.clone(), action)
		isValid = self.validRadii(nextRadii)

		if isValid:
			self.radii = nextRadii
		else:
			self.radii = prevRadii

		# self.TSCS, self.RMS = self.getMetric(self.radii)
		self.counter += 1/100
		done = False
		if int(self.counter.item()) == 1:
			done = True
		return done
		# reward = self.getReward(self.RMS, isValid)
		# nextState = torch.cat([self.radii, self.TSCS, self.RMS, time], dim=-1)
		# return nextState, reward, done

if __name__ == '__main__':
	config = [1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0]
	env = RadiusEnv(
		k0amax=0.45,
		k0amin=0.35,
		nFreq=11, 
		config=config)

	actionRange = (env.maxRadii - env.minRadii)/20

	import imageio
	writer = imageio.get_writer('radii2.mp4', format='mp4', mode='I', fps=15)

	env.reset()
	done = False
	while not done:
		action = np.random.uniform(
			-actionRange, 
			actionRange, 
			size=(1, env.nCyl))

		done = env.step(action)
		img = env.getIMG(env.radii)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()

