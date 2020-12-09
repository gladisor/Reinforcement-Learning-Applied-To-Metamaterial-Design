from env import TSCSEnv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io
import torch
import matlab

class RadiusEnv(TSCSEnv):
	"""docstring for RadiusEnv"""
	def __init__(self, k0amax, k0amin, nfreq, config):
		super(RadiusEnv, self).__init__(int(len(config)), k0amax, k0amin, nfreq)
		## Configuration of cylinders to be optimized
		self.config = torch.tensor(config)
		self.radii = None
		self.c_pv = None
		self.rho_sh = None

		## Configuration of cylinders to be held fixed
		self.center_config = torch.tensor([[0.0, 0.0]])
		self.center_radii = torch.ones(1, self.center_config.shape[0])
		self.center_c_p = torch.tensor([5480.0])
		self.center_rho_sh = torch.tensor([8850.0])

		## Radius range
		self.minRadii = 0.2
		self.maxRadii = 1.0
		self.minDistanceBetween = 0.1

		## c_p and rho_sh ranges
		self.min_c_p = 4.0e3
		self.max_c_p = 6.0e3

		self.min_rho_sh = 2.0e3
		self.max_rho_sh = 9.0e3

	def validRadii(self, radii):
		withinBounds = False
		overlap = False

		all_radii = torch.cat([radii, self.center_radii], dim=-1)
		if (self.minRadii <= all_radii).all() and (all_radii <= self.maxRadii).all():
			withinBounds = True

			coords = torch.cat([self.config, self.center_config], dim=0)
			for i in range(coords.shape[0]):
				for j in range(coords.shape[0]):
					if i != j:
						x1, y1 = coords[i]
						r1 = all_radii[0, i]
						x2, y2 = coords[j]
						r2 = all_radii[0, j]
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

		coords = torch.cat([self.config, self.center_config], dim=0)
		all_radii = torch.cat([radii, self.center_radii], dim=-1)
		for cyl in range(coords.shape[0]):
			ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=all_radii[0, cyl]))

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
		coords = torch.cat([self.config, self.center_config], dim=0)
		xM = matlab.double(coords.tolist())
		all_radii = torch.cat([self.radii, self.center_radii], dim=-1)
		av = matlab.double(all_radii.tolist())

		c_pv = torch.ones(coords.shape[0])*self.center_c_p
		c_pv = matlab.double(c_pv.tolist())
		rho_shv = torch.ones(coords.shape[0])*self.center_rho_sh
		rho_shv = matlab.double(rho_shv.tolist())
		tscs = self.eng.getMetric_thinShells_radii_material(xM, av, c_pv, rho_shv, self.k0amax, self.k0amin, self.nfreq)
		# tscs = self.eng.getMetric_RigidCylinder_radii(xM, av, self.k0amax, self.k0amin, self.nfreq)
		tscs = torch.tensor(tscs).T
		rms = tscs.pow(2).mean().sqrt().view(1,1)
		return tscs, rms

	def reset(self):
		self.radii = self.getInitialRadii()
		self.TSCS, self.RMS = self.getMetric(self.radii)
		self.counter = torch.tensor([[0.0]])
		state = torch.cat([self.radii, self.TSCS, self.RMS, self.counter], dim=-1)
		return state

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

		self.TSCS, self.RMS = self.getMetric(self.radii)
		self.counter += 1/100
		done = False
		if int(self.counter.item()) == 1:
			done = True

		reward = self.getReward(self.RMS, isValid)
		nextState = torch.cat([self.radii, self.TSCS, self.RMS, self.counter], dim=-1)
		return nextState, reward, done

if __name__ == '__main__':
	from circlePoints import rtpairs

	r = [3.5]
	n = [10]
	circle = rtpairs(r, n)

	env = RadiusEnv(
		k0amax=0.45,
		k0amin=0.35,
		nfreq=11, 
		config=circle)

	actionRange = (env.maxRadii - env.minRadii)/20

	import imageio
	writer = imageio.get_writer('continuousRadii.mp4', format='mp4', mode='I', fps=15)

	env.reset()
	done = False
	while not done:
		action = np.random.uniform(
			-actionRange, 
			actionRange, 
			size=(1, env.nCyl))

		nextState, reward, done = env.step(action)
		img = env.getIMG(env.radii)
		print(env.getMetric(env.radii))
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()

