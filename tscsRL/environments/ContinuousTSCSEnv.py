from tscsRL.environments.BaseTSCSEnv import BaseTSCSEnv
import imageio
import numpy as np

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

if __name__ == '__main__':
	name = 'continuousTestVideo'
	writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)

	nCyl = 4
	kMax = 0.45
	kMin = 0.35
	nFreq = 11
	stepSize = 0.5

	env = ContinuousTSCSEnv(nCyl, kMax, kMin, nFreq, stepSize)

	state = env.reset()
	done = False

	while not done:
		action = np.random.normal(0, 1, size=(1, env.action_space))
		action = np.clip(action, -env.stepSize, env.stepSize)

		nextState, reward, done = env.step(action)
		print(reward)
		state = nextState

		img = env.getIMG(env.config)
		writer.append_data(img.view(env.img_dim).numpy())

	writer.close()