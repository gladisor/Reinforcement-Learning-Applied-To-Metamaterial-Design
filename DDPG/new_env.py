import matlab.engine
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from torchvision import transforms
from PIL import Image
import io
import numpy as np


## 4 Cylinder TSCS
class TSCSEnv():
    """docstring for TSCSEnv"""

    def __init__(self, params):
        # Matlab interface
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('TSCS')
        # Hyperparameters
        self.nCyl = params.NCYL
        self.M = matlab.double([self.nCyl])
        self.k0amax = matlab.double([params.KMAX])
        self.k0amin = matlab.double([params.KMIN])
        self.nfreq = matlab.double([params.NFREQ])

        # State variables
        self.config = None
        self.TSCS = None
        self.RMS = None
        self.img = None
        self.counter = None

        # Image transform
        self.img_dim = 50
        self.transform = transforms.Compose([
            transforms.Resize((self.img_dim, self.img_dim)),
            transforms.Grayscale(),
            transforms.ToTensor()])

    def getPenalty(self, config):
        penalty = 0.0
        coords = config.view(self.nCyl, 2)
        for i in range(self.nCyl):
            for j in range(self.nCyl):
                if i != j:
                    x1, y1 = coords[i]
                    x2, y2 = coords[j]
                    d = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if d <= 2.1:
                        penalty += d
                    d_out = torch.sqrt(x1 ** 2 + y1 ** 2) - 5
                    if d_out > 0:
                        penalty += d_out
        return float(penalty)

    def validConfig(self, config):
        """
        Checks if config is within bounds and does not overlap cylinders
        """
        withinBounds = False
        overlap = False
        if (-5 < config).all() and (config < 5).all():
            withinBounds = True

            coords = config.view(self.nCyl, 2)
            for i in range(self.nCyl):
                for j in range(self.nCyl):
                    if i != j:
                        x1, y1 = coords[i]
                        x2, y2 = coords[j]
                        d = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                        if d <= 2.1:
                            overlap = True
        return withinBounds and not overlap

    def getConfig(self):
        """
        Generates a configuration which is within bounds
        and not overlaping cylinders
        """
        while True:
            config = torch.FloatTensor(1, 2 * self.nCyl).uniform_(-5, 5)
            if self.validConfig(config):
                break
        return config

    def getMetric(self, config):
        x = self.eng.transpose(matlab.double(*config.tolist()))
        tscs = self.eng.getMetric(x, self.M, self.k0amax, self.k0amin, self.nfreq)
        tscs = torch.tensor(tscs).T
        rms = tscs.pow(2).mean().sqrt().view(1, 1)
        return tscs, rms

    # def getTSCS(self, config):
    # 	## Gets tscs of configuration from matlab
    # 	tscs = self.eng.getTSCS4CYL(*config.squeeze(0).tolist())
    # 	return torch.tensor(tscs).T

    # def getRMS(self, config):
    # 	## Gets rms of configuration from matlab
    # 	rms = self.eng.getRMS4CYL(*config.squeeze(0).tolist())
    # 	return torch.tensor([[rms]])

    def getIMG(self, config):
        """
        Produces tensor image of configuration
        """
        # To avoind repeatedly displaying figure while training ImageDDPG
        plt.ioff()
        # Generate figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('equal')
        ax.set_xlim(xmin=-6, xmax=6)
        ax.set_ylim(ymin=-6, ymax=6)
        ax.grid()

        coords = config.view(self.nCyl, 2)
        for cyl in range(self.nCyl):
            ax.add_artist(Circle((coords[cyl, 0], coords[cyl, 1]), radius=1))

        # Convert to tensor
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)

        # Apply series of transformations
        X = self.transform(im)

        buf.close()
        plt.close(fig)
        return X.unsqueeze(0)

    def getTime(self):
        return self.counter / 100

    def getReward(self, RMS, penalty):
        """
        Computes reward based on change in scattering
        proporitional to how close it is to zero
        """
        reward = 1 / RMS.item() - 1000 * np.log(penalty+1)

        return reward

    def reset(self):
        """
        Generates starting config and calculates its tscs
        """
        self.config = self.getConfig()
        self.TSCS, self.RMS = self.getMetric(self.config)
        self.img = self.getIMG(self.config)

        self.counter = torch.tensor([[0.0]])
        time = self.getTime()
        state = torch.cat([self.config, self.TSCS, self.RMS, time], dim=-1).float()
        return state

    def getNextConfig(self, config, action):
        """
        Applys action to config
        """
        return config + action

    def step(self, action):
        """
        If the config after applying the action is not valid
        we revert back to previous state and give negative reward
        otherwise, reward is calculated by the change in scattering
        """
        nextConfig = self.getNextConfig(self.config, action)
        penalty = self.getPenalty(nextConfig)

        self.config = nextConfig
        self.img = self.getIMG(self.config)

        self.TSCS, self.RMS = self.getMetric(self.config)
        # self.TSCS = self.getTSCS(self.config)
        # self.RMS = self.getRMS(self.config)
        self.counter += 1
        time = self.getTime()

        reward = self.getReward(self.RMS, penalty)
        nextState = torch.cat([self.config.double(), self.TSCS.double(), self.RMS.double(), time.double()],
                              dim=-1).float()
        return nextState, reward
