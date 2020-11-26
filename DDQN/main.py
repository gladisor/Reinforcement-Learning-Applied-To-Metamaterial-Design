from env import TSCSEnv
from agent import Agent
from models import CylinderCoordConv, CylinderNet
import torch
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
	import wandb
	## Hyperparameters
	GAMMA = 0.9
	EPS = 1
	EPS_END = 0.1
	EPS_DECAY_STEPS = 2500
	TARGET_UPDATE = 10
	MEMORY_SIZE = 1_000_000
	BATCH_SIZE = 64
	LR = 0.0005
	MOMENTUM = 0.9
	NUM_EPISODES = 3000
	EPISODE_LEN = 100
	H_SIZE = 128
	N_HIDDEN = 1

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

	wandb.init(project='tscs-ddqn')

	## Creating agent object with parameters
	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY_STEPS, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	# Defining models
	agent.Qp = CylinderNet(
		env.nCyl * 2 + env.F + 2,
		H_SIZE, 
		N_HIDDEN,
		env.nCyl * 4).cuda()

	agent.Qt = CylinderNet(
		env.nCyl * 2 + env.F + 2,
		H_SIZE,
		N_HIDDEN,
		env.nCyl * 4).cuda()

	agent.Qt.eval()
	agent.opt = torch.optim.SGD(
		agent.Qp.parameters(),
		lr=LR,
		momentum=MOMENTUM)
	
	agent.Qt.load_state_dict(agent.Qp.state_dict())
	agent.nActions = nCyl * 2

	## This is the holder for transition data
	agent.Transition = namedtuple('Transition', 
		('c','tscs','rms','time',
		'a','r',
		'c_','tscs_','rms_','time_','done'))


	step = 0

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
			f'I:{initial}, '\
			f'L:{lowest}, '\
			f'F:{current}, '\
			f'Score:{episode_reward}, '\
			f'Eps:{agent.eps}')

		wandb.log({
			'lowest':lowest,
			'reward':episode_reward,
			'epsilon':agent.eps})

		## Save models
		if episode % 100 == 0:
			torch.save(agent.Qp.state_dict(), f'ddqn.pt')