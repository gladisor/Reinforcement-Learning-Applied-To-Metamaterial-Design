import sys
sys.path.insert(0, '../DDPG')
from radiusEnv import RadiusEnv
import numpy as np

class DiscreteRadiusEnv(RadiusEnv):
	def __init__(self, k0amax, k0amin, nfreq, config):
		super(DiscreteRadiusEnv, self).__init__(k0amax, k0amin, nfreq, config)
		self.actionRange = (self.maxRadii - self.minRadii)/20

		self.action_space = self.nCyl * 2
		self.observation_space = self.nCyl + self.F + 2

	def getNextRadii(self, radii, action):
		cyl = int(action/2)
		direction = action % 2
		
		if direction == 1:
			radii[0, cyl] += self.actionRange
		else:
			radii[0, cyl] -= self.actionRange
		return radii

if __name__ == '__main__':
	from circlePoints import rtpairs

	r = [3.1]
	n = [9]

	circle = rtpairs(r, n)

	env = DiscreteRadiusEnv(
		k0amax=0.45,
		k0amin=0.35,
		nfreq=11, 
		config=circle)

	import imageio
	writer = imageio.get_writer('descreteRadii.mp4', format='mp4', mode='I', fps=15)

	print(f'nCyl: {env.nCyl}')

	env.reset()
	done = False
	while not done:
		action = np.random.randint(env.nCyl * 2)
		nextState, reward, done = env.step(action)
		print(reward)
		img = env.getIMG(env.radii)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	writer.close()

