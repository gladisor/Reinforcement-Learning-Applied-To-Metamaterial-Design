import numpy as np
from env import TSCSEnv
import matplotlib.pyplot as plt
from agent import Agent
import torch as T

def get_min_config(env, agent):
	state = env.reset()
	done = False

	results = {
		'initial': env.config,
		'rms': env.RMS.item(),
		'tscs': env.TSCS,
		'config': env.config}

	while not done:
		action = agent.select_action(state)
		
		nextState, reward, done = env.step(action)
		state = nextState

		if env.RMS.item() < results['rms']:
			results['rms'] = env.RMS.item()
			results['tscs'] = env.TSCS
			results['config'] = env.config
	return results

if __name__ == '__main__':
	env = TSCSEnv(4, 0.45, 0.35, 11, 0.5)
	params = {
		'inSize': env.nCyl * 2 + env.F + 2,
	    'hSize': 128,
	    'nHidden': 2,
	    'nActions': env.nCyl * 4,
	    'lr': 5e-4,
	    'gamma': 0.90,
	    'epsEnd': 0.10,
	    'epsDecaySteps': 2_000,
	    'memorySize': 1_000_000,
	    'batchSize': 256,
	    'tau': 0.005,
	    'num_random_episodes':10,
	    'name': '2Cyl256BatchSize'}

	agent = Agent(params)
	agent.Qt.load_state_dict(T.load('savedModels/4cyl0.45-0.35LargerNet.pt'))
	agent.eps = 0.1

	lowest, batch = [], []
	for _ in range(100):
		results = get_min_config(env, agent)
		lowest.append(results['rms'])
		batch.append(results)
		print(results)

	print('BEST:')
	idx = lowest.index(min(lowest))
	best = batch[idx]
	print(best)

	plt.imshow(env.getIMG(best['config']).view(50, 50))
	plt.show()
