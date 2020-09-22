import numpy as np

class ActionNoise(object):
    def reset(self):
        pass

class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    noise = OrnsteinUhlenbeckActionNoise(
        mu=np.zeros(1),
        sigma=np.ones(1)*0.2)

    log = []
    noise.reset()
    for i in range(100):
        log.append(noise())


    plt.plot(log)
    plt.show()

    noise = OrnsteinUhlenbeckActionNoise(
    mu=np.zeros(5),
    sigma=np.ones(5)*0.2)

    noise.reset()
    for i in range(100):
        print(noise())
