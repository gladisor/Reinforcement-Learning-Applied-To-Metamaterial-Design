from env import TSCSEnv
from train import Agent
import torch
import torch.nn as nn
from tqdm import tqdm
from collections import namedtuple

class CylinderNet(nn.Module):
	def __init__(self):
		super(CylinderNet, self).__init__()
		self.fc1 = nn.Linear(19, 100)
		self.fc2 = nn.Linear(100, 100)
		self.v = nn.Linear(100, 1)
		self.adv = nn.Linear(100, 16)

	def forward(self, s):
		x = torch.cat([s[0], s[1]], dim=-1)
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		a = self.adv(x)
		q = self.v(x) + a - a.mean(-1, keepdim=True)
		return q

if __name__ == '__main__':
	GAMMA = 0.99
	EPS = 1
	EPS_END = 0.05
	EPS_DECAY = 0.99
	TARGET_UPDATE = 1500
	MEMORY_SIZE = 10_000
	BATCH_SIZE = 32
	LR = 0.0005
	NUM_EPISODES = 300

	agent = Agent(
		GAMMA, EPS, EPS_END, EPS_DECAY, 
		MEMORY_SIZE, BATCH_SIZE, LR)

	agent.Qp, agent.Qt = CylinderNet(), CylinderNet()
	agent.opt = torch.optim.RMSprop(agent.Qp.parameters(), lr=LR)
	agent.Qt.eval()

	env = TSCSEnv()

	Transition = namedtuple('Transition', ('c','tscs','a','r','c_','tscs_','done'))

	from tqdm import tqdm
	step = 0
	running_reward = 0
	hist = []
	for episode in range(NUM_EPISODES):
		episode_reward = 0
		state = state = env.reset()
		initial = state[1].mean().item()
		for t in tqdm(range(500)):
			action = agent.select_action(state)
			nextState, reward, done = env.step(action)
			episode_reward += reward
			step += 1

			action = torch.tensor([action])
			reward = torch.tensor([reward]).float()
			done = torch.tensor([done])
			e = Transition(
				state[0], state[1], 
				action, reward, 
				nextState[0], nextState[1], done)

			agent.memory.push(e)
			loss = agent.optimize_model(e)

			state = nextState
			if step % TARGET_UPDATE == 0:
				agent.Qt.load_state_dict(agent.Qp.state_dict())

			if done:
				break

		final = state[1].mean().item()
		print(f'#: {episode}, Initial: {round(initial, 4)}, Final: {round(final, 4)}, Score: {episode_reward}, Eps: {round(agent.eps, 2)}')
		hist.append(episode_reward)
		agent.finish_episode()
	
	del agent.memory.memory[:]
	env.close()
	plt.plot(hist)
	plt.show()

	torch.save(agent.Qt.state_dict(), 'm.pt')
