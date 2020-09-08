# import gym
import torch
from models import CylinderCoordConv, CylinderNet, DQN
from env import TSCSEnv
import matplotlib.pyplot as plt
import numpy as np
import random

dqn = CylinderNet(useCuda=False)
dqn.load_state_dict(torch.load('model.pt'))
# env = gym.make('LunarLander-v2')
env = TSCSEnv()

plt.ion()
fig = plt.figure()
ax = fig.add_subplot()

state = env.reset()
img = env.getIMG(state[0])
myobj = ax.imshow(img.view(50, 50))

for t in range(100):
	if random.random() > 0.1:
		## Exploit
		with torch.no_grad():
			action = torch.argmax(dqn(state), dim=-1).item()
	else:
		## Explore
		action = np.random.randint(16)

	state, reward, done = env.step(action)

	img = env.getIMG(state[0])
	myobj.set_data(img.view(50, 50))
	fig.canvas.draw()
	fig.canvas.flush_events()
	plt.pause(0.01)

	print(f"RMS: {round(state[2].item(), 2)}")
	print(f"Reward: {reward}")
	if done:
		break