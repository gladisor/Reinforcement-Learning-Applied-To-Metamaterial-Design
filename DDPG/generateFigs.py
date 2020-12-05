from env import TSCSEnv
import torch
import matplotlib.pyplot as plt
import numpy as np

nCyl = 2
k0amax = 0.45
k0amin = 0.35
nfreq = 11
env = TSCSEnv(nCyl=nCyl, k0amax=k0amax, k0amin=k0amin, nfreq=nfreq)
initial = torch.tensor([[ 4.5641, -2.7947,  2.8730,  0.4883]])
ddqn = torch.tensor([[ 3.5641,  1.2053, -0.1270,  0.9883]])
ddpg = torch.tensor([[ 4.1739, -3.8139,  0.3324, -3.7284]])

initialTSCS, _ = env.getMetric(initial)
ddqnTSCS, _ = env.getMetric(ddqn)
ddpgTSCS, _ = env.getMetric(ddpg)

print(initialTSCS)
print(ddqnTSCS)
print(ddpgTSCS)

# plt.plot(initialTSCS[0], label='initial')
# plt.plot(ddqnTSCS[0], label='ddqn')
# plt.plot(ddpgTSCS[0], label='ddpg')
# ticks = np.round(np.linspace(k0amin, k0amax, nfreq), 2)
# plt.xticks(np.arange(nfreq), ticks)
# plt.title(f'TSCS vs ka for {nCyl} Scatterers')
# plt.xlabel('ka')
# plt.ylabel('TSCS')
# plt.legend()
# plt.show()

img = env.getIMG(ddpg)
plt.imshow(img.view(env.img_dim, env.img_dim), cmap='gray')
plt.axis('off')
plt.show()