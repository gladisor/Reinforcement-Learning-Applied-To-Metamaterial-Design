from agent import Agent
from descreteRadiusEnv import DiscreteRadiusEnv
from dqn_models import DQN
import torch
from collections import namedtuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from circlePoints import rtpairs

if __name__ == '__main__':
	## Hyperparameters
	GAMMA = 0.9
	EPS = 1
	EPS_END = 0.05
	EPS_DECAY_STEPS = 3000
	TARGET_UPDATE = 10
	MEMORY_SIZE = 1_000_000
	BATCH_SIZE = 64
	LR = 0.0005
	MOMENTUM = 0.9
	NUM_EPISODES = 5000
	EPISODE_LEN = 100
	H_SIZE = 128
	N_HIDDEN = 2

	k0amax=0.45
	k0amin=0.35
	nfreq=11

	r = [2.1, 4.1]
	n = [4, 8]

	circle = rtpairs(r, n)
	## Creating environment object
	env = DiscreteRadiusEnv(
		k0amax=k0amax,
		k0amin=k0amin,
		nfreq=nfreq, 
		config=circle)

	wandb.init(project='tscs-ddpg-radii')

	## Creating agent object with parameters
	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY_STEPS, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	# Defining models
	agent.Qp = DQN(
		env.observation_space,
		H_SIZE, 
		N_HIDDEN,
		env.action_space)

	agent.Qt = DQN(
		env.observation_space,
		H_SIZE,
		N_HIDDEN,
		env.action_space)

	agent.Qt.eval()
	agent.opt = torch.optim.SGD(
		agent.Qp.parameters(),
		lr=LR,
		momentum=MOMENTUM)
	
	agent.Qt.load_state_dict(agent.Qp.state_dict())
	agent.nActions = env.action_space

	## This is the holder for transition data
	agent.Transition = namedtuple('Transition', ('s','a','r','s_','done'))

	step = 0

	for episode in range(NUM_EPISODES):
		## Reset reward and env
		episode_reward = 0
		state = env.reset()

		## Record initial scattering
		initial = env.RMS.item()
		lowest = initial
		for t in tqdm(range(EPISODE_LEN + 1)):
			## Select action, observe nextState & reward
			action = agent.select_action(state)
			nextState, reward, done = env.step(action)

			episode_reward += reward
			step += 1

			# Update current lowest
			current = env.RMS.item()
			if current < lowest:
				lowest = current

			action = torch.tensor([[action]])
			reward = torch.tensor([[reward]]).float()
			done = torch.tensor([done])
			e = agent.Transition(state, action, reward, nextState, done)

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
			path = 'radiiResults/12cyl/'
			torch.save(agent.Qp.state_dict(), path + f'Qpolicy{episode}.pt')
			torch.save(agent.Qp.state_dict(), path + f'Qtarget{episode}.pt')