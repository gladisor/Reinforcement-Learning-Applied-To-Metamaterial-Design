import gym
import torch
from models import CylinderCoordConv
from env import TSCSEnv

dqn = CylinderCoordConv()
env = TSCSEnv()

state = env.reset()
for t in range(200):
	env.render()
	action = torch.argmax(dqn(state), dim=-1).item()
	print(action)
	state, reward, done = env.step(action)
	if done:
		break