from env import TSCSEnv
from agent import Agent
import torch as T
import wandb

env = TSCSEnv(4, 0.45, 0.35, 11, 0.5)

params = {
	'inSize': env.nCyl * 2 + env.F + 2,
    'hSize': 128,
    'nHidden': 2,
    'nActions': env.nCyl * 4,
    'lr': 5e-4,
    'gamma': 0.90,
    'epsEnd': 0.10,
    'epsDecaySteps': 3_000,
    'memorySize': 1_000_000,
    'batchSize': 64,
    'tau': 0.005,
    'num_random_episodes':0,
    'name': '4cyl0.45-0.35LargerNet'}

agent = Agent(params)

wandb.init(project='tscs-ddqn', config=params)

for episode in range(20_000):
	episode_reward = 0
	state = env.reset()

	initial = env.RMS.item()
	lowest = initial
	done = False
	while not done:
		action = agent.select_action(state)
		nextState, reward, done = env.step(action)

		episode_reward += reward

		current = env.RMS.item()
		if current < lowest:
			lowest = current

		e = agent.Transition(
			state, 
			T.tensor([[action]]), 
			T.tensor([[reward]]), 
			nextState, 
			T.LongTensor([[done]]))

		## Add most recent transition to memory and update model
		agent.memory.push(e)
		agent.optimize_model()
		state = nextState

	print(f'#: {episode}, reward: {episode_reward}, I: {initial}, L: {lowest}, epsilon: {agent.eps}')
	wandb.log({'reward': episode_reward, 'initial': initial, 'lowest': lowest, 'epsilon':agent.eps})

	if episode > params['num_random_episodes']:
		agent.decay_epsilon()

	if episode % 100 == 0:
		name = params['name']
		T.save(agent.Qt.state_dict(), f'savedModels/{name}.pt')
