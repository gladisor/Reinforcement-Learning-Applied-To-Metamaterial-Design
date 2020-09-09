import gym
from env import TSCSEnv
from agent import Agent
from models import CylinderCoordConv, CylinderNet
import torch
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
	## Hyperparameters
	GAMMA = 0.9
	EPS = 1
	EPS_END = 0.1
	EPS_DECAY = 0.998
	TARGET_UPDATE = 10
	MEMORY_SIZE = 100_000
	BATCH_SIZE = 64
	LR = 0.0005
	MOMENTUM = 0.9
	NUM_EPISODES = 30_000
	EPISODE_LEN = 100
	USE_CUDA = True
	H_SIZE = 128
	N_HIDDEN = 3

	## Creating agent object with parameters
	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR, useCuda=USE_CUDA)

	# Defining models
	agent.Qp = CylinderNet(useCuda=USE_CUDA, H_SIZE, N_HIDDEN).cuda()
	agent.Qt = CylinderNet(useCuda=USE_CUDA, H_SIZE, N_HIDDEN).cuda()
	agent.Qt.eval()
	agent.opt = torch.optim.SGD(
		agent.Qp.parameters(),
		lr=LR,
		momentum=MOMENTUM)

	agent.Qt.load_state_dict(agent.Qp.state_dict())
	agent.nActions = 16

	## This is the holder for transition data
	agent.Transition = namedtuple('Transition', 
		('c','tscs','rms','time',
		'a','r',
		'c_','tscs_','rms_','time_','done'))

	## Creating environment object
	env = TSCSEnv()

	step = 0
	writer = SummaryWriter(
		f'grid_search/{GAMMA}gamma-SGD-{MOMENTUM}momentum-{TARGET_UPDATE}targetupdate-3hidden')

	for episode in range(NUM_EPISODES):
		## Reset reward and env
		episode_reward = 0
		state = env.reset()

		## Record initial scattering
		initial = state[1].mean().item()
		lowest = initial
		for t in tqdm(range(EPISODE_LEN)):
			## Select action, observe nextState & reward
			action = agent.select_action(state)
			nextState, reward, done = env.step(action)

			episode_reward += reward
			step += 1

			# Update current lowest
			current = state[1].mean().item()
			if current < lowest:
				lowest = current

			if t == EPISODE_LEN:
				done = True

			action = torch.tensor([[action]])
			reward = torch.tensor([[reward]]).float()
			done = torch.tensor([done])
			e = agent.Transition(*state, action, reward, *nextState, done)

			## Add most recent transition to memory and update model
			agent.memory.push(e)
			loss = agent.optimize_model()
			state = nextState

			## Copy policy weights over to target net
			if step % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())
				step = 0

			## End episode if terminal state
			if done:
				break

		agent.decay_epsilon()
		print(
			f'#:{episode}, '\
			f'I:{round(initial, 2)}, '\
			f'Lowest:{round(lowest, 2)}, '\
			f'F:{round(current, 2)}, '\
			f'Score:{round(episode_reward, 2)}, '\
			f'Eps:{round(agent.eps, 2)}')

		writer.add_scalar('train/score', episode_reward, episode)
		writer.add_scalar('train/lowest', lowest, episode)
	torch.save(agent.Qt.state_dict(), 'model.pt')
