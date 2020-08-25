import gym
import torch
from train import DQN

dqn = DQN()
dqn.load_state_dict(torch.load('m.pt'))
env = gym.make('LunarLander-v2')
state = torch.tensor([env.reset()]).float()
for t in range(1000):
	env.render()
	action = torch.argmax(dqn(state), dim=-1).item()
	state, reward, done, _ = env.step(action)
	if done:
		break
	state = torch.tensor([state]).float()

env.close()