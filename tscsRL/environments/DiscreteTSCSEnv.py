from tscsRL.environments.BaseTSCSEnv import BaseTSCSEnv
import imageio
import numpy as np

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
	name = 'discreteTestVideo'
	writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)

	nCyl = 4
	kMax = 0.45
	kMin = 0.35
	nFreq = 11
	stepSize = 0.5

	env = DiscreteTSCSEnv(nCyl, kMax, kMin, nFreq, stepSize)

	state = env.reset()
	done = False

	while not done:
		action = np.random.randint(env.action_space)

		nextState, reward, done = env.step(action)
		print(reward)
		state = nextState

		img = env.getIMG(env.config)
		writer.append_data(img.view(env.img_dim).numpy())

	writer.close()




