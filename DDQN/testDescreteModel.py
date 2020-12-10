import torch
import imageio
from circlePoints import rtpairs
import random
import matplotlib.pyplot as plt
from descreteRadiusEnv import DiscreteRadiusEnv
from dqn_models import DQN
import numpy as np

H_SIZE = 128
N_HIDDEN = 2

k0amax=0.45
k0amin=0.35
nfreq=11

r = [2.1, 4.1]
n = [4, 8]
circle = rtpairs(r, n)

env = DiscreteRadiusEnv(
	k0amax=k0amax,
	k0amin=k0amin,
	nfreq=nfreq, 
	config=circle)

# Defining models
dqn = DQN(
	env.observation_space,
	H_SIZE, 
	N_HIDDEN,
	env.action_space)

dqn.load_state_dict(torch.load('radiiResults/12cyl/Qpolicy3600.pt'))

state = env.reset()

results = {
	'radii': [],
	'rms': [],
	'tscs': []}

results['radii'].append(env.radii)
results['rms'].append(env.RMS)
results['tscs'].append(env.TSCS)

writer = imageio.get_writer('test2ring.mp4', format='mp4', mode='I', fps=15)
img = env.getIMG(env.radii)
writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

done = False
while not done:
	if random.random() > 0.05:
		## Exploit
		with torch.no_grad():
			action = torch.argmax(dqn(state), dim=-1).item()
	else:
		## Explore
		action = np.random.randint(env.action_space)

	nextState, reward, done = env.step(action)

	print(reward)

	results['radii'].append(env.radii)
	results['rms'].append(env.RMS)
	results['tscs'].append(env.TSCS)
	state = nextState

	img = env.getIMG(env.radii)
	writer.append_data(img.view(env.img_dim, env.img_dim).numpy())

writer.close()

minIdx = results['rms'].index(min(results['rms']))

initialRMS = results['rms'][0]
initialRadii = results['radii'][0]
optimalRadii = results['radii'][minIdx]
optimalRMS = results['rms'][minIdx]
optimalTSCS = results['tscs'][minIdx]

print(f'Initial RMS: {initialRMS}')
print(f'Initial Radii: {initialRadii}')
print(f'Optimal Radii: {optimalRadii}')
print(f'Optimal RMS: {optimalRMS}')
print(f'Optimal TSCS: {optimalTSCS}')

plt.imshow(env.getIMG(results['radii'][minIdx]).view(env.img_dim, env.img_dim))
plt.show()