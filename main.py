from env import TSCSEnv
from agent import Agent
from models import CylinderNet, CylinderCoordConv

import torch
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
	GAMMA = 0.99
	EPS = 1
	EPS_END = 0.05
	EPS_DECAY = 0.99
	TARGET_UPDATE = 1000 ## Default 1500
	MEMORY_SIZE = 10_000 ## Default 10_000
	BATCH_SIZE = 64
	LR = 0.0005
	NUM_EPISODES = 5000
	EPISODE_LEN = 200

	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	agent.Qp, agent.Qt = CylinderCoordConv().cuda(), CylinderCoordConv().cuda()
	agent.opt = torch.optim.RMSprop(agent.Qp.parameters(), lr=LR)
	agent.Qt.eval()
	agent.Qt.load_state_dict(agent.Qp.state_dict())
	
	agent.Transition = namedtuple('Transition', 
		('c','tscs','rms','img',
		'a','r',
		'c_','tscs_','rms_','img_','done'))

	env = TSCSEnv()

	step = 0
	smoothed_reward = 0
	hist = {'score':[],'smooth_score':[],'length':[]}
	for episode in range(NUM_EPISODES):
		## Reset reward and env
		episode_reward = 0
		state = env.reset()

		## Record initial scattering
		initial = state[1].mean().item()
		for t in tqdm(range(EPISODE_LEN)):
			## Select action, observe nextState & reward
			action = agent.select_action(state)
			nextState, reward, done = env.step(action)

			episode_reward += reward
			step += 1

			if t == EPISODE_LEN - 1:
				done = True

			action = torch.tensor([[action]])
			reward = torch.tensor([[reward]]).float()
			done = torch.tensor([done])
			e = agent.Transition(
				*state, action, reward, *nextState, done)

			agent.memory.push(e)
			loss = agent.optimize_model(e)

			state = nextState
			if step % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())
				step = 0

			if done:
				break

		smoothed_reward = smoothed_reward * 0.9 + episode_reward * 0.1
		final = state[1].mean().item()
		print(f'#: {episode}, I: {round(initial, 2)}, F: {round(final, 2)}, Score: {round(episode_reward, 2)}, Eps: {round(agent.eps, 2)}')
		hist['score'].append(episode_reward)
		hist['smooth_score'].append(smoothed_reward)
		hist['length'].append(t)
		agent.finish_episode()
	
	del agent.memory.memory[:]
	plt.plot(hist['score'])
	plt.plot(hist['smooth_score'])
	plt.show()

	plt.plot(hist['length'])
	plt.show()

	torch.save(agent.Qt.state_dict(), 're_training.pt')
