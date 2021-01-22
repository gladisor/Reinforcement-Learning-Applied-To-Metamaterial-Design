from tscsRL.environments.RadiiTSCSEnv import BaseRadiiTSCSEnv

import torch
import numpy as np
import gym
import matlab
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib import colors

class BaseMaterialTSCSEnv(BaseRadiiTSCSEnv):
	"""docstring for MaterialTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, design_material):
		super(BaseMaterialTSCSEnv, self).__init__(kMax, kMin, nFreq)
		self.design_material = design_material

		## rho_sh and c_p ranges
		self.min_rho_sh = 2.0e3
		self.max_rho_sh = 9.0e3
		self.rho_sh_stepSize = (self.max_rho_sh - self.min_rho_sh)/20

		self.min_c_p = 4.0e3
		self.max_c_p = 6.0e3
		self.c_p_stepSize = (self.max_c_p - self.min_c_p)/20

		## properties of nickle
		self.nickle_rho_sh = 8850.0
		self.nickle_c_p = 5480.0

		## setting materials of central object
		self.center_rho_sh = torch.ones(1, self.center_M) * self.nickle_rho_sh
		self.center_c_p = torch.ones(1, self.center_M) * self.nickle_c_p

		## setting static radii
		self.radii = torch.ones(1, self.design_M) * 0.7
		self.all_radii = torch.cat([self.radii, self.center_radii], dim=-1)

		self.color_mapping = plt.get_cmap('viridis')

		## setting material properties vectors
		if self.design_material == 'rho_sh':
			self.rho_shv = None
			self.c_pv = torch.ones(1, self.design_M) * self.nickle_c_p
		elif self.design_material == 'c_p':
			self.c_pv = None
			self.rho_shv = torch.ones(1, self.design_M) * self.nickle_rho_sh
		else:
			print('Unrecognized material property. Try: (rho_sh, c_p)')

		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, self.design_M * 1 + nFreq + 2))

	def validRho_shv(self, rho_shv):
		withinRange = False
		if (self.min_rho_sh <= rho_shv).all() and (rho_shv <= self.max_rho_sh).all():
			withinRange = True
		return withinRange

	def validC_pv(self, c_pv):
		withinRange = False
		if (self.min_c_p <= c_pv).all() and (c_pv <= self.max_c_p).all():
			withinRange = True
		return withinRange

	def validMaterial(self, rho_shv, c_pv):
		valid = False
		if self.validRho_shv(rho_shv) and self.validC_pv(c_pv):
			valid = True
		return valid

	def getRho_shv(self):
		return torch.FloatTensor(1, self.design_M).uniform_(self.min_rho_sh, self.max_rho_sh)

	def getC_pv(self):
		return torch.FloatTensor(1, self.design_M).uniform_(self.min_c_p, self.max_c_p)

	def setMetric(self, rho_shv, c_pv):
		x = self.all_config.view(1, 2 * self.nCyl)
		x = self.eng.transpose(matlab.double(*x.tolist()))
		av = self.eng.transpose(matlab.double(*self.all_radii.tolist()))

		all_c_pv = torch.cat([c_pv, self.center_c_p], dim=-1)
		all_c_pv = self.eng.transpose(matlab.double(*all_c_pv.tolist()))
		all_rho_shv = torch.cat([rho_shv, self.center_rho_sh], dim=-1)
		all_rho_shv = self.eng.transpose(matlab.double(*all_rho_shv.tolist()))

		tscs = self.eng.getMetric_thinShells_radii_material(x, self.M, av, all_c_pv, all_rho_shv, self.kMax, self.kMin, self.nFreq)
		self.TSCS = torch.tensor(tscs).T
		self.RMS = self.TSCS.pow(2).mean().sqrt().view(1,1)

	def getState(self):
		if self.design_material == 'rho_sh':
			state = torch.cat([self.rho_shv, self.TSCS, self.RMS, self.counter], dim=-1).float()
		elif self.design_material == 'c_p':
			state = torch.cat([self.c_pv, self.TSCS, self.RMS, self.counter], dim=-1).float()
		else:
			pass
		return state

	def reset(self):
		if self.design_material == 'rho_sh':
			self.rho_shv = self.getRho_shv()
		elif self.design_material == 'c_p':
			self.c_pv = self.getC_pv()
		else:
			print('Invalid design material')

		self.counter = torch.tensor([[0.0]])
		self.setMetric(self.rho_shv, self.c_pv)
		state = self.getState()

		## Log initial scattering at beginning of episode and reset score
		self.info['initial'] = self.RMS.item()
		self.info['lowest'] = self.info['initial']
		self.info['final'] = None
		self.info['score'] = 0
		return state

	def getNextMaterial(self, rho_shv, c_pv, action):
		raise NotImplementedError

	def step(self, action):
		"""
		Updates the state of the environment with action. Returns next state, reward, done.
		"""
		prev_rho_shv = self.rho_shv.clone()
		prev_c_pv = self.c_pv.clone()
		next_rho_shv, next_c_pv = self.getNextMaterial(self.rho_shv.clone(), self.c_pv.clone(), action)
		isValid = self.validMaterial(next_rho_shv, next_c_pv)

		if isValid:
			self.rho_shv = next_rho_shv
			self.c_pv = next_c_pv
		else: ## Invalid next state, do not change state variables
			self.rho_shv = prev_rho_shv
			self.c_pv = prev_c_pv

		self.setMetric(self.rho_shv, self.c_pv)
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

	def getIMG(self, rho_shv, c_pv):
		"""
		Produces tensor image of state
		"""
		## Generate figure
		fig, ax = plt.subplots(figsize=(6, 6))
		ax.axis('equal')
		ax.set_xlim(xmin=-self.grid_size - 1, xmax=self.grid_size + 1)
		ax.set_ylim(ymin=-self.grid_size - 1, ymax=self.grid_size + 1)
		ax.grid()

		all_rho_shv = torch.cat([rho_shv, self.center_rho_sh], dim=-1)
		all_c_pv = torch.cat([c_pv, self.center_c_p], dim=-1)

		for cyl in range(self.nCyl):
			x, y = self.all_config[cyl]
			r = self.all_radii[0, cyl]
			rho_sh = (all_rho_shv[0, cyl].item() - self.min_rho_sh) / (self.max_rho_sh - self.min_rho_sh)
			c_p = (all_c_pv[0, cyl].item() - self.min_c_p) / (self.max_c_p - self.min_c_p)
			if self.design_material == 'rho_sh':
				color = self.color_mapping(rho_sh)
			elif self.design_material == 'c_p':
				color = self.color_mapping(c_p)
			ax.add_artist(Circle((x, y), radius=r, color=color))

		fig.canvas.draw()

		width, height = fig.get_size_inches() * fig.get_dpi()
		img = np.fromstring(fig.canvas.tostring_rgb(), dtype='uint8')
		img = img.reshape(int(height), int(width), 3)

		plt.close(fig)
		return img

class ContinuousMaterialTSCSEnv(BaseMaterialTSCSEnv):
	"""docstring for ContinuousMaterialTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, design_material):
		super(ContinuousMaterialTSCSEnv, self).__init__(kMax, kMin, nFreq, design_material)

		if self.design_material == 'rho_sh':
			self.action_space = gym.spaces.Box(
				low=-self.rho_sh_stepSize,
				high=self.rho_sh_stepSize,
				shape=(1, self.design_M))
		elif self.design_material == 'c_p':
			self.action_space = gym.spaces.Box(
				low=-self.c_p_stepSize,
				high=self.c_p_stepSize,
				shape=(1, self.design_M))
		else:
			self.action_space = None

	def getNextMaterial(self, rho_shv, c_pv, action):
		if self.design_material == 'rho_sh':
			rho_shv =  rho_shv + action
		elif self.design_material == 'c_p':
			c_pv =  c_pv + action

		return rho_shv, c_pv

class DiscreteMaterialTSCSEnv(BaseMaterialTSCSEnv):
	"""docstring for DiscreteMaterialTSCSEnv"""
	def __init__(self, kMax, kMin, nFreq, design_material):
		super(DiscreteMaterialTSCSEnv, self).__init__(kMax, kMin, nFreq, design_material)

		self.action_space = gym.spaces.Discrete(2 * self.design_M)

	def getNextMaterial(self, rho_shv, c_pv, action):
		cyl = int(action / 2)
		direction = action % 2

		print(cyl, direction)

		if self.design_material == 'rho_sh':
			if direction == 0:
				rho_shv[0, cyl] += self.rho_sh_stepSize
			elif direction == 1:
				rho_shv[0, cyl] += -self.rho_sh_stepSize
		elif self.design_material == 'c_p':
			if direction == 0:
				c_pv[0, cyl] += self.c_p_stepSize
			elif direction == 1:
				c_pv[0, cyl] += -self.c_p_stepSize
		return rho_shv, c_pv

if __name__ == '__main__':
	from tscsRL.environments.MaterialTSCSEnv import DiscreteMaterialTSCSEnv
	import imageio
	import matplotlib.pyplot as plt

	writer = imageio.get_writer('material.mp4', format='mp4', mode='I', fps=15)

	env = DiscreteMaterialTSCSEnv(0.45, 0.35, 11, 'rho_sh')

	state = env.reset()

	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)

		img = env.getIMG(env.rho_shv, env.c_pv)
		writer.append_data(img)

	writer.close()