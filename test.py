import gym
import torch
from models import CylinderCoordConv, DQN
# from env import TSCSEnv

dqn = DQN()
dqn.load_state_dict(torch.load('model.pt'))
env = gym.make('LunarLander-v2')
# env = TSCSEnv()

state = env.reset()
for t in range(1000):
	env.render()
	state = torch.tensor([state]).float()
	action = torch.argmax(dqn(state), dim=-1).item()
	print(action)
	state, reward, done, _ = env.step(action)
	if done:
		break