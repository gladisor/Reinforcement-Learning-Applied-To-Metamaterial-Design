# import gym
import torch
from models import CylinderCoordConv, CylinderNet, DQN
from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
import random
from models import CylinderNet
import imageio

def evaluate(agent, env):
	# state = env.reset()
	env.config = torch.tensor([[-4.5320,  3.2800, -3.8886, -3.5087, -4.2241,  0.7828,  1.1836,  3.8953]])
	env.TSCS, env.RMS = env.getMetric(env.config)
	env.counter = torch.tensor([[0.0]])
	time = env.getTime()
	state = torch.cat([env.config, env.TSCS, env.RMS, time], dim=-1).float()

	results = {
		'config': [],
		'rms': [],
		'tscs': []}
	results['config'].append(env.config)
	results['rms'].append(env.RMS)
	results['tscs'].append(env.TSCS)

	writer = imageio.get_writer('video.mp4', format='mp4', mode='I', fps=15)
	img = env.getIMG(env.config)
	writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

	for t in range(100):
		if random.random() > 0.1:
			## Exploit
			with torch.no_grad():
				action = torch.argmax(dqn(state), dim=-1).item()
		else:
			## Explore
			action = np.random.randint(4 * nCyl)

		nextState, reward, done = env.step(action)

		results['config'].append(env.config)
		results['rms'].append(env.RMS)
		results['tscs'].append(env.TSCS)
		state = nextState

		img = env.getIMG(env.config)
		writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

		if done:
			break

	writer.close()
	return results

if __name__ == '__main__':
	nCyl=4
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

	dqn.load_state_dict(torch.load('saved_models/ddqn8000.pt', map_location=torch.device('cpu')))

	results = evaluate(dqn, env)
	minIdx = results['rms'].index(min(results['rms']))

	initialRMS = results['rms'][0]
	print(f'Initial RMS: {initialRMS}')
	print(results['config'][minIdx])
	print(results['rms'][minIdx])
	print(results['tscs'][minIdx])


