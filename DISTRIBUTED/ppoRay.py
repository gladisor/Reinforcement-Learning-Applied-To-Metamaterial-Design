import ray
from ray.rllib.agents import ppo
from ray import tune
from env import DistributedTSCSEnv
import wandb

if __name__ == '__main__':
	config = ppo.DEFAULT_CONFIG.copy()
	config['framework'] = 'torch'

	## Paralellism
	config['num_workers'] = 16
	config['num_envs_per_worker'] = 1
	config['num_cpus_per_worker'] = .4
	config['num_cpus_for_driver'] = 1
	config['num_gpus'] = 1

	## Env hyperparameters
	config['env'] = DistributedTSCSEnv
	config['env_config'] = {
		'nCyl':4,
		'k0amax':0.5,
		'k0amin':0.3,
		'nFreq':11,
		'actionRange':0.2,
		'episodeLength':100}

	## RL hyperparameters
	config['gamma'] = 0.90

	## DL hyperparameters
	config['rollout_fragment_length'] = config['env_config']['episodeLength']
	config['train_batch_size'] = 4000
	config['sgd_minibatch_size'] = 64
	config['num_sgd_iter'] = 30
	config['lr'] = 5e-5
	config['model']['fcnet_hiddens'] = [256] * 2
	config['model']['fcnet_activation'] = 'relu'

	## Evaluation
	config['evaluation_interval'] = 1
	config['evaluation_num_episodes'] = 5

	ray.init()
	wandb.init(project='TSCS-PPO', config=config)

	agent = ppo.PPOTrainer(config)

	for train_cycle in range(100):
		result = agent.train()
		data = {
			'episode_reward_mean': result['episode_reward_mean'],
			'eval_reward_mean': result['evaluation']['episode_reward_mean'],
			'episodes_this_iter': result['episodes_this_iter']}

		print(data)
		wandb.log(data)
		if train_cycle % 10 == 0:
			agent.save('third_run')