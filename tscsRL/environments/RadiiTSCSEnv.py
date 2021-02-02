from tscsRL.environments.TSCSEnv import BaseTSCSEnv
from tscsRL.utils import rtpairs

import torch
import matlab
import gym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io

class BaseRadiiTSCSEnv(BaseTSCSEnv):
	"""docstring for BaseRadiiTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, ring_radii=[3.1, 5.2], nCyl_ring=[9, 10], core_radius=1.6):
		## Define the object in the center which is static
		self.center_config = torch.tensor([[0.0, 0.0]])
		self.center_M = self.center_config.shape[0]
		self.core_radius = core_radius
		self.center_radii = torch.ones(1, self.center_M) * self.core_radius

		self.ring_radii = ring_radii
		self.nCyl_ring = nCyl_ring
		self.design_M = sum(self.nCyl_ring)

		## Radius range
		self.min_radii = 0.2
		self.max_radii = 1.0
		self.min_distance = 0.1

		## Creating variables to store total number of cylinders and radius adjustment
		nCyl = self.design_M + self.center_M
		stepSize = (self.max_radii - self.min_radii)/20

		## Invoke superconstructor 
		super(BaseRadiiTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)
		self.grid_size = 10.0

		## State information
		self.config = torch.tensor(rtpairs(self.ring_radii, self.nCyl_ring))
		self.all_config = torch.cat([self.config, self.center_config])
		self.radii = None

		## Material properties for all cylinders
		self.c_p = 5480.0
		self.rho_sh = 8850.0
		material_vector = torch.ones(1, self.nCyl)
		self.c_pv = material_vector * self.c_p
		self.rho_shv = material_vector * self.rho_sh
		## Converting to matlab
		self.c_pv = matlab.double(self.c_pv.tolist())
		self.rho_shv = matlab.double(self.rho_shv.tolist())

		## Observation
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, self.design_M + nFreq + 2))

	def getParams(self):
		env_params = {
			'nCyl': self.nCyl,
			'kMax': np.array(self.kMax).item(),
			'kMin': np.array(self.kMin).item(),
			'nFreq': self.F,
			'ep_len': self.ep_len,
			'grid_size': self.grid_size,
			'stepSize': self.stepSize,
			'ring_radii': self.ring_radii,
			'nCyl_ring': self.nCyl_ring,
			'core_radius': self.core_radius,
		}
		return env_params

	def validRadii(self, radii):
		withinRange = False
		overlap = False

		if (self.min_radii <= radii).all() and (radii <= self.max_radii).all():
			withinRange = True
			## Concatenating variable radii with static center radius
			all_radii = torch.cat([radii, self.center_radii], dim=-1)
			for i in range(self.nCyl):
				for j in range(self.nCyl):
					if i != j:
						## Getting data about cylinder i
						x1, y1 = self.all_config[i]
						r1 = all_radii[0, i]

						## Getting data about cylinder j
						x2, y2 = self.all_config[j]
						r2 = all_radii[0, j]

						## Calculating distance between i and j
						d = torch.sqrt((x2 - x1)**2 + (y2 - y1)**2)

						if r1 + r2 + self.min_distance >= d:
							overlap = True
		return withinRange and not overlap

	def getRadii(self):
		while True:
			radii = torch.FloatTensor(1, self.design_M).uniform_(self.min_radii, self.max_radii)
			if self.validRadii(radii):
				break
		return radii

	def setMetric(self, radii):
		x = self.all_config.view(1, 2 * self.nCyl)
		x = self.eng.transpose(matlab.double(*x.tolist()))
		all_radii = torch.cat([radii, self.center_radii], dim=-1)
		av = self.eng.transpose(matlab.double(*all_radii.tolist()))

		tscs = self.eng.getMetric_RigidCyliders_radii(x, self.M, av, self.kMax, self.kMin, self.nFreq)
		# tscs = self.eng.getMetric_thinShells_radii_material(x, self.M, av, self.c_pv, self.rho_shv, self.kMax, self.kMin, self.nFreq)
		self.TSCS = torch.tensor(tscs).T
		self.RMS = self.TSCS.pow(2).mean().sqrt().view(1,1)

	def getIMG(self, radii):
		"""
		Produces tensor image of state
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-self.grid_size - 1, xmax=self.grid_size + 1)
		ax.set_ylim(ymin=-self.grid_size - 1, ymax=self.grid_size + 1)
		ax.grid()

		all_radii = torch.cat([radii, self.center_radii], dim=-1)
		for cyl in range(self.nCyl):
			x, y = self.all_config[cyl]
			ax.add_artist(Circle((x, y), radius=all_radii[0, cyl]))

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

	def renderIMG(self, radii):
		"""
		Produces tensor image of state
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-self.grid_size - 1, xmax=self.grid_size + 1)
		ax.set_ylim(ymin=-self.grid_size - 1, ymax=self.grid_size + 1)
		ax.grid()

		all_radii = torch.cat([radii, self.center_radii], dim=-1)
		for cyl in range(self.nCyl):
			x, y = self.all_config[cyl]
			ax.add_artist(Circle((x, y), radius=all_radii[0, cyl]))

		plt.show()

	def getState(self):
		state = torch.cat([self.radii, self.TSCS, self.RMS, self.counter], dim=-1).float()
		return state

	def reset(self):
		"""
		Generates starting radii and calculates its tscs
		"""
		self.radii = self.getRadii()
		self.counter = torch.tensor([[0.0]])
		self.setMetric(self.radii)
		state = self.getState()

		## Log initial scattering at beginning of episode and reset score
		self.info['initial'] = self.RMS.item()
		self.info['lowest'] = self.info['initial']
		self.info['final'] = None
		self.info['score'] = 0
		return state

	def getNextRadii(self, radii, action):
		raise NotImplementedError

	def step(self, action):
		"""
		Updates the state of the environment with action. Returns next state, reward, done.
		"""
		prevRadii = self.radii.clone()
		nextRadii = self.getNextRadii(self.radii.clone(), action)
		isValid = self.validRadii(nextRadii)

		if isValid:
			self.radii = nextRadii
		else: ## Invalid next state, do not change state variables
			self.radii = prevRadii

		self.setMetric(self.radii)
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

class ContinuousRadiiTSCSEnv(BaseRadiiTSCSEnv):
	"""docstring for ContinuousRadiiTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, ring_radii=[3.1, 5.2], nCyl_ring=[9, 10], core_radius=1.6):
		super(ContinuousRadiiTSCSEnv, self).__init__(kMax, kMin, nFreq, ring_radii, nCyl_ring, core_radius)
		
		## Action space
		self.action_space = gym.spaces.Box(
			low=-self.stepSize,
			high=self.stepSize,
			shape=(1, self.design_M))

	def getNextRadii(self, radii, action):
		return radii + action

class DiscreteRadiiTSCSEnv(BaseRadiiTSCSEnv):
	"""docstring for DiscreteRadiiTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, ring_radii=[3.1, 5.2], nCyl_ring=[9, 10], core_radius=1.6):
		super(DiscreteRadiiTSCSEnv, self).__init__(kMax, kMin, nFreq, ring_radii, nCyl_ring, core_radius)
		
		## Action space
		self.action_space = gym.spaces.Discrete(2 * self.design_M)

	def getNextRadii(self, radii, action):
		cyl = int(action / 2)
		direction = action % 2

		if direction == 0:
			radii[0, cyl] += self.stepSize
		elif direction == 1:
			radii[0, cyl] += -self.stepSize
		else:
			print('Unrecognized action')
		return radii

if __name__ == '__main__':
	env = ContinuousRadiiTSCSEnv(
		kMax=0.45,
		kMin=0.35,
		nFreq=11)
		# ring_radii=[4.7, 7, 9.5],
		# nCyl_ring=[6, 7, 10],
		# core_radius=3.2)

	env.radii = torch.tensor([[0.3997, 0.3258, 0.2163, 0.3209, 0.5080, 0.4866, 0.2933, 0.2925, 0.3569, 0.7287, 0.5647, 0.3234, 0.3024, 0.4325, 0.7464, 0.4650, 0.3630, 0.3009, 0.5665]])
	env.setMetric(env.radii)
	print(env.TSCS)
	print(env.RMS)

	# state = env.reset()

	# import imageio

	# writer = imageio.get_writer('continuous_radii.mp4', format='mp4', mode='I', fps=15)

	# done = False
	# while not done:
	# 	img = env.getIMG(env.radii)
	# 	writer.append_data(img.view(env.img_dim).numpy())

	# 	print(env.RMS, done)
	# 	action = env.action_space.sample()
	# 	state, reward, done, info = env.step(action)

	# writer.close()
