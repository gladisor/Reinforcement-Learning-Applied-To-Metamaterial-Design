# import gym
import torch
from models import CylinderCoordConv, CylinderNet, DQN
from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
import random
from models import CylinderNet

nCyl=3
k0amax=0.45
k0amin=0.35
nfreq=11
STEP_SIZE = 0.5

## Creating environment object
env = TSCSEnv(
	nCyl=nCyl, 
	k0amax=k0amax, 
	k0amin=k0amin, 
	nfreq=nfreq, 
	stepSize=STEP_SIZE)

dqn = CylinderNet(
	env.nCyl * 2 + env.F + 2,
	128, 
	1,
	env.nCyl * 4)

dqn.load_state_dict(torch.load('3cylLongTrain.pt'))


def evaluate(env, agent):
	state = env.reset()

	results = {
		'rms': env.RMS,
		'config': env.config,
		'tscs': env.TSCS
	}

	for t in range(100):
		if random.random() > 0.1:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(dqn(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(2 * nCyl)

		state, reward, done = env.step(action)

		if env.RMS < results['rms']:
			results['rms'] = env.RMS
			results['config'] = env.config
			results['tscs'] = env.TSCS

		if done:
			break
	return results

for _ in range(10):
	results = evaluate(env, dqn)
	print(results)

# plt.imshow(env.getIMG(results['config']).view(50, 50))
# plt.show()