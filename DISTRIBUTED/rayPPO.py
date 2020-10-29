import ray
from ray.rllib.agents import ppo
from ray import tune
from env import DistributedTSCSEnv
from ray.tune.integration.wandb import WandbLogger

if __name__ == '__main__':
	config = ppo.DEFAULT_CONFIG.copy()
	config['framework'] = 'torch'

	## Paralellism
	config['num_workers'] = 10
	config['num_cpus_per_worker'] = 0.3
	config['num_cpus_for_driver'] = 1
	config['num_gpus'] = 1

	## Env hyperparameters
	config['env'] = DistributedTSCSEnv
	config['env_config'] = {
		'nCyl':4,
		'k0amax':0.5,
		'k0amin':0.3,
		'nFreq':11,
		'actionRange':0.5,
		'episodeLength':100}

	config['wandb']: {
		'project': 'TSCS-PPO',
		'api_key': '4977efc1b4ac78735811330325031d96a51d4010',
		'log_config': True}

	## RL hyperparameters
	config['gamma'] = 0.90

	## DL hyperparameters
	config['rollout_fragment_length'] = config['env_config']['episodeLength']
	config['train_batch_size'] = 4000
	config['sgd_minibatch_size'] = 128
	config['num_sgd_iter'] = 30
	config['lr'] = 5e-5
	config['model']['fcnet_hiddens'] = [256] * 2
	config['model']['fcnet_activation'] = 'relu'


	## Stopping criteria
	stop = {
		'timesteps_total':1_000_000}

	tune.run(
		ppo.PPOTrainer,
		config=config,
		stop=stop,
		loggers=[WandbLogger])