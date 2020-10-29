import ray
from ray.rllib.agents import ddpg
from ray import tune
from env import DistributedTSCSEnv
import torch

if __name__ == '__main__':
	config = ddpg.DEFAULT_CONFIG.copy()
	config['framework'] = 'torch'

	## Paralellisim 
	config['num_workers'] = 1
	config['num_gpus'] = 1
	
	## RL hyperparameters
	config['gamma'] = 0.90
	config['exploration_config']['ou_sigma'] = 0.5
	config['exploration_config']['initial_scale'] = 1.0
	config['exploration_config']['final_scale'] = 0.1
	config['exploration_config']['scale_timesteps'] = 100_000
	config['n_step'] = 1

	## DL hyperparameters
	config['train_batch_size'] = 256
	config['actor_hiddens'] = [128] * 2
	config['actor_hidden_activation'] = 'relu'
	config['actor_lr'] = 0.0001

	config['critic_hiddens'] = [128] * 8
	config['critic_hidden_activation'] = 'relu'
	config['critic_lr'] = 0.001
	config['tau'] = 0.002

	## Env hyperparameters
	config['env'] = DistributedTSCSEnv
	config['env_config'] = {
		'nCyl':4,
		'k0amax':0.5,
		'k0amin':0.3,
		'nFreq':11,
		'actionRange':0.5,
		'episodeLength':100}

	## Stopping criteria
	stop = {
		'timesteps_total':1_000_000}

	ray.init()
	agent = ddpg.DDPGTrainer(config)

	env = DistributedTSCSEnv(config['env_config'])
	state = env.reset()
	print(agent.compute_action(state))

	# N = 10
	# episode_data = []
	# for n in range(N):
	#     result = agent.train()
	#     episode = {
	#         'n': n,
	#         'episode_reward_mean': result['episode_reward_mean']}
	#     episode_data.append(episode)
	#     print(f'episode: {n}')

	# tune.run(
	#     ddpg.DDPGTrainer,
	#     config=config,
	#     stop=stop)