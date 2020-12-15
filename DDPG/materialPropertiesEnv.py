from radiusEnv import RadiusEnv
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import matlab
import matplotlib.cm as cm
import io
from PIL import Image

class MaterialPropertiesEnv(RadiusEnv):
	"""docstring for MaterialPropertiesEnv"""
	def __init__(self, k0amax, k0amin, nfreq, config, radii):
		super(MaterialPropertiesEnv, self).__init__(k0amax, k0amin, nfreq, config)
		self.radii = torch.tensor(radii)

		## c_p and rho_sh ranges
		self.min_c_p = 4.0e3
		self.max_c_p = 6.0e3

		self.min_rho_sh = 2.0e3
		self.max_rho_sh = 9.0e3

		self.c_pv = None
		self.rho_shv = torch.ones(self.config.shape[0] + self.center_config.shape[0])*self.center_rho_sh

	def validMaterials(self, c_pv):
		withinBounds = False
		overlap = False

		if (self.min_c_p <= c_pv).all() and (c_pv <= self.max_c_p).all():
			withinBounds = True

		return withinBounds

	def getInitialMaterials(self):
		while True:
			c_pv = torch.FloatTensor(self.nCyl).uniform_(self.min_c_p, self.max_c_p)
			if self.validMaterials(c_pv):
				break
		return c_pv

	def getIMG(self, c_pv):
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
		all_radii = torch.cat([self.radii, self.center_radii], dim=-1)
		all_c_pv = torch.cat([self.c_pv, self.center_c_p])
		for cyl in range(coords.shape[0]):
			ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=all_radii[0, cyl], color=cm.hot(all_c_pv[cyl].item()/4000)))

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

	def getMetric(self, c_pv):
		coords = torch.cat([self.config, self.center_config], dim=0)
		all_radii = torch.cat([self.radii, self.center_radii], dim=-1)

		all_c_pv = torch.cat([c_pv, self.center_c_p])
		all_rho_shv = torch.cat([self.rho_shv, self.center_rho_sh])

		xM = matlab.double(coords.tolist())
		av = matlab.double(all_radii.tolist())
		c_pv = matlab.double(all_c_pv.tolist())
		rho_shv = matlab.double(all_rho_shv.tolist())

		tscs = self.eng.getMetric_thinShells_radii_material(xM, av, c_pv, rho_shv, self.k0amax, self.k0amin, self.nfreq)

		tscs = torch.tensor(tscs).T
		rms = tscs.pow(2).mean().sqrt().view(1,1)
		return tscs, rms

	def reset(self):
		self.c_pv = self.getInitialMaterials()
		self.TSCS, self.RMS = self.getMetric(self.c_pv)
		self.counter = torch.tensor([[0.0]])
		state = torch.cat([self.c_pv.unsqueeze(0), self.TSCS, self.RMS, self.counter], dim=-1)
		return state

	def getNextMaterial(self, c_pv, action):
		return c_pv + action

	def step(self, action):
		prevMaterial = self.c_pv.clone()
		nextMaterial = self.getNextMaterial(self.c_pv.clone(), action)
		isValid = self.validMaterials(nextMaterial)

		if isValid:
			self.c_pv = nextMaterial
		else:
			self.c_p = prevMaterial

		self.TSCS, self.RMS = self.getMetric(self.c_pv)
		self.counter += 1/100
		done = False
		if int(self.counter.item()) == 1:
			done = True

		reward = self.getReward(self.RMS, isValid)
		nextState = torch.cat([self.c_pv.unsqueeze(0), self.TSCS, self.RMS, self.counter], dim=-1)
		return nextState, reward, done

if __name__ == '__main__':
	from circlePoints import rtpairs

	r = [3.1]
	n = [9]
	circle = rtpairs(r, n)
	radii = [[0.4062, 0.3450, 0.9399, 0.3591, 0.3141, 0.2118, 0.5394, 0.9254, 0.4142]]

	env = MaterialPropertiesEnv(
		k0amax=0.45,
		k0amin=0.35,
		nfreq=11, 
		config=circle,
		radii=radii)

	actionRange = (env.max_c_p - env.min_c_p)/20

	import imageio
	writer = imageio.get_writer('testMaterials.mp4', format='mp4', mode='I', fps=15)

	env.reset()
	done = False
	while not done:
		action = np.random.uniform(
			-actionRange, 
			actionRange, 
			size=(env.nCyl))

		nextState, reward, done = env.step(action)
		print(env.c_pv)
		img = env.getIMG(env.c_pv)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()

