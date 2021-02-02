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
	# 	"""
	# 	Computes reward based on change in scattering 
	# 	proporitional to how close it is to zero
	# 	"""
	# 	reward = -RMS.item()
	# 	if not isValid:
	# 		reward += -1.0 / self.nCyl
	# 	return reward

	# def getReward(self, RMS, isValid):
	# 	gradient_penalty = torch.linalg.norm(self.gradient)

	# 	## Penalizing high gradient
	# 	reward = -RMS - gradient_penalty
	# 	if not isValid:
	# 		reward += -1.0
	# 	return reward.item()

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
		nCyl=10,
		kMax=0.45,
		kMin=0.35,
		nFreq=11,
		stepSize=0.5)

	env.config = torch.tensor([[-0.7228, -0.0434, -1.8131, -4.9530,  1.7860,  2.5667, -2.3883,  3.1135, 1.2655, -0.7409,  0.8594, -4.9058,  3.8888,  3.3243, -0.2760,  3.1022, -2.0707, -2.5604,  3.1582, -3.1052]]).double()

	env.setMetric(env.config)

	print(env.RMS)

	# state = env.reset()
	# print(state)
	# print(state.shape)

	# action = np.random.normal(0, 1, size=(1, env.action_space))

	# nextState, reward, done, info = env.step(action)
	# print(nextState)
	# print(reward)
	# print(done)
	# print(info)

	# env = DiscreteGradientTSCSEnv(
	# 	nCyl=2,
	# 	kMax=0.45,
	# 	kMin=0.35,
	# 	nFreq=11,
	# 	stepSize=0.5)

	# state = env.reset()
	# print(state)
	# print(state.shape)

	# action = np.array([[np.random.randint(env.action_space)]])

	# nextState, reward, done, info = env.step(action)
	# print(nextState)
	# print(reward)
	# print(done)
	# print(info)