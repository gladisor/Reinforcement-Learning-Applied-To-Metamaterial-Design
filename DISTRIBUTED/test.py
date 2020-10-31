import ray
from ray.rllib.agents import ppo
from ray import tune
from env import DistributedTSCSEnv
import wandb
import numpy as np

if __name__ == '__main__':
	config = ppo.DEFAULT_CONFIG.copy()
	config['framework'] = 'torch'

	config['num_workers'] = 1

	## Env hyperparameters
	config['env'] = DistributedTSCSEnv
	config['env_config'] = {
		'nCyl':4,
		'k0amax':0.5,
		'k0amin':0.3,
		'nFreq':11,
		'actionRange':0.2,
		'episodeLength':100}

	## Model hyperparameters
	config['model']['fcnet_hiddens'] = [256] * 2
	config['model']['fcnet_activation'] = 'relu'
	
	ray.init()
	agent = ppo.PPOTrainer(config)
	agent.restore('third_run/checkpoint_91/checkpoint-91')
	env = DistributedTSCSEnv(config['env_config'])

	state = env.reset()
	done = False

	while not done:
		action = agent.compute_action(state)
		state, reward, done, info = env.step(action)
		print(f'Reward: {reward}, RMS: {env.RMS.item()}')

