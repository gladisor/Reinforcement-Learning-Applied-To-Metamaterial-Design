from tscsRL.environments.TSCSEnv import BaseTSCSEnv, ContinuousTSCSEnv, DiscreteTSCSEnv
import matlab
import torch
import numpy as np
import gym

class BaseGradientTSCSEnv(BaseTSCSEnv):
	"""docstring for BaseGradientTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		super(BaseGradientTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)

		## New state variable
		self.gradient = None

		## Observation space changes from 2 * nCyl to 4 * nCyl due to additional gradient info
		self.observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(1, 4 * nCyl + nFreq + 2))

	def setMetric(self, config):
		x = self.eng.transpose(matlab.double(*config.tolist()))
		tscs, grad = self.eng.getMetric_Rigid_Gradient(x, self.M, self.kMax, self.kMin, self.nFreq, nargout=2)
		self.TSCS = torch.tensor(tscs).T
		self.RMS = self.TSCS.pow(2).mean().sqrt().view(1,1)
		self.gradient = torch.tensor(grad).T

	def getState(self):
		state = torch.cat([self.config, self.TSCS, self.RMS, self.gradient, self.counter], dim=-1).float()
		return state

	# def getReward(self, RMS, isValid):
	# 	gradient_penalty = np.linalg.norm(np.array(self.gradient))
	# 	gradient_penalty = torch.tensor(gradient_penalty)
	# 	## Penalizing high gradient
	# 	reward = -RMS - gradient_penalty
	# 	if not isValid:
	# 		reward += -1.0
	# 	return reward.item()

	## Modification for Multiple cylinders
	def getReward(self, RMS, inValid):
		# New reward function. Invalid
		return -RMS.item() - inValid/5

	def checkIsValid(self, x1, y1, i, config):
		withinBounds = False
		overlap = False
		config = torch.cat([config[0:i], config[i+1:]])

		if (self.grid_size > abs(x1)) and (self.grid_size > abs(y1)):
			withinBounds = True

		for i in range(self.nCyl-1):
			x2, y2 = config[i]
			d = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
			if d <= 2.1:
				overlap = True

		return withinBounds and not overlap

	def getNextConfig(self, config, actions):
		config = config.view(self.nCyl, 2)
		nextConfig = config
		actions = actions.view(self.nCyl, 2)
		inValid = self.nCyl

		for i in range(self.nCyl):
			x1, y1 = config[i] + actions[i]

			isValid = self.checkIsValid(x1, y1, i, nextConfig)

			if isValid:
				nextConfig[i] = torch.tensor([x1, y1])
				inValid -= 1

		return nextConfig.view(1, 2*self.nCyl), inValid

	def step(self, action):
		self.config, inValid = self.getNextConfig(self.config, action)

		self.setMetric(self.config)
		self.counter += 1 / self.ep_len
		nextState = self.getState()

		reward = self.getReward(self.RMS, inValid)
		self.info['score'] += reward

		done = False
		if int(self.counter.item()) == 1:
			done = True

		# Update current lowest scatter
		current = self.RMS.item()
		if current < self.info['lowest']:
			self.info['lowest'] = current

		self.info['final'] = current

		return nextState, reward, done, self.info, inValid




class ContinuousGradientTSCSEnv(BaseGradientTSCSEnv, ContinuousTSCSEnv):
	"""docstring for ContinuousGradientTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		super(ContinuousGradientTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)

class DiscreteGradientTSCSEnv(BaseGradientTSCSEnv, DiscreteTSCSEnv):
	"""docstring for DiscreteGradientTSCSEnv"""
	def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
		super(DiscreteGradientTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)


if __name__ == '__main__':
	import numpy as np
	env = ContinuousGradientTSCSEnv(
		nCyl=2,
		kMax=0.45,
		kMin=0.35,
		nFreq=11,
		stepSize=0.5)

	state = env.reset()
	print(state)
	print(state.shape)

	action = np.random.normal(0, 1, size=(1, env.action_space))

	nextState, reward, done, info = env.step(action)
	print(nextState)
	print(reward)
	print(done)
	print(info)

	env = DiscreteGradientTSCSEnv(
		nCyl=2,
		kMax=0.45,
		kMin=0.35,
		nFreq=11,
		stepSize=0.5)

	state = env.reset()
	print(state)
	print(state.shape)

	action = np.array([[np.random.randint(env.action_space)]])

	nextState, reward, done, info = env.step(action)
	print(nextState)
	print(reward)
	print(done)
	print(info)