from tscsRL.environments.TSCSEnv import BaseTSCSEnv, ContinuousTSCSEnv
import matlab
import torch
import numpy as np
import gym


class ImageTSCSEnv(ContinuousTSCSEnv):
    """docstring for BaseGradientTSCSEnv"""

    def __init__(self, nCyl, kMax, kMin, nFreq, stepSize):
        super(ImageTSCSEnv, self).__init__(nCyl, kMax, kMin, nFreq, stepSize)

        ## New state variable
        self.gradient = None
        self.img = None

        ## Observation space changes from 2 * nCyl to 4 * nCyl due to additional gradient info
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1, 4 * nCyl + nFreq + 2))

    def setMetric(self, config):
        x = self.eng.transpose(matlab.double(*config.tolist()))
        tscs, grad = self.eng.getMetric_Rigid_Gradient(x, self.M, self.kMax, self.kMin, self.nFreq, nargout=2)
        self.TSCS = torch.tensor(tscs).T
        self.RMS = self.TSCS.pow(2).mean().sqrt().view(1, 1)
        self.gradient = torch.tensor(grad).T

    def getState(self):
        state = torch.cat([self.config, self.TSCS, self.RMS, self.gradient, self.counter], dim=-1).float()
        return state

    def reset(self):
        """
        Generates starting config and calculates its tscs
        """
        self.config = self.getConfig()
        self.counter = torch.tensor([[0.0]])
        self.setMetric(self.config)
        self.img = self.getIMG(self.config)
        state = self.getState()

        ## Log initial scattering at beginning of episode and reset score
        self.info['initial'] = self.RMS.item()
        self.info['lowest'] = self.info['initial']
        self.info['final'] = None
        self.info['score'] = 0
        return state

    def step(self, action):
        """
        Updates the state of the environment with action. Returns next state, reward, done.
        """
        prevConfig = self.config.clone()
        nextConfig = self.getNextConfig(self.config.clone(), action)
        isValid = self.validConfig(nextConfig)

        if isValid:
            self.config = nextConfig
        else:  ## Invalid next state, do not change state variables
            self.config = prevConfig

        self.setMetric(self.config)
        self.counter += 1 / self.ep_len
        self.img = self.getIMG(self.config)
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

if __name__ == '__main__':
    import numpy as np

    env = ImageTSCSEnv(
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
