from env import TSCSEnv
import matlab.engine
import torch

env = TSCSEnv()
state = env.reset()


M = matlab.double([4])
k0amax = matlab.double([0.5])
k0amin = matlab.double([0.3])
nfreq = matlab.double([11])

x = env.eng.transpose(matlab.double(*env.config.tolist()))

tscs = env.eng.getMetric(x, M, k0amax, k0amin, nfreq)
tscs = torch.tensor(tscs).T
rms = tscs.pow(2).mean().sqrt().view(1,1)

print(rms)
print(tscs)