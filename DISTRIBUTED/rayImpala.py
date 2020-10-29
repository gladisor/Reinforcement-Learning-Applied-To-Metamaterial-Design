import ray
from ray.rllib.agents import impala
from ray import tune
from env import DistributedTSCSEnv
import torch

if __name__ == '__main__':
	config = impala.DEFAULT_CONFIG.copy()
	config['framework'] = 'torch'

	## Paralellisim 
	config['num_workers'] = 5
	config['num_cpus_per_worker'] = 1
	config['num_gpus'] = 1
	
	## RL hyperparameters
	config['gamma'] = 0.90
	# config['exploration_config'] = {
	# 	'type': 'OrnsteinUhlenbeckNoise',
	# 	'random_timesteps': 0,
	# 	'ou_base_scale': 0.1,
	# 	'ou_theta': 0.15,
	# 	'ou_sigma': 0.5,
	# 	'initial_scale': 1.0,
	# 	'final_scale': 0.2,
	# 	'scale_timesteps': 100}

	## DL hyperparameters
	config['train_batch_size'] = 10000
	config['rollout_fragment_length'] = 100

	config['model']['fcnet_hiddens'] = [256] * 4
	config['model']['fcnet_activation'] = 'relu'
	config['lr'] = 0.0005

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
	agent = impala.ImpalaTrainer(config)

	N = 10
	for n in range(N):
	    result = agent.train()
	    episode = {
	    	'gpu_util_percent':result['perf']['gpu_util_percent0'],
	    	'ram_util_percent':result['perf']['ram_util_percent'],
	    	'episode_reward_mean':result['episode_reward_mean'],
	    	'num_episodes':len(result['hist_stats']['episode_reward']),
	    	# 'ou_sigma':result['config']['exploration_config']['ou_base_scale']
	    	}
	    print(episode)

	# tune.run(
	#     impala.ImpalaTrainer,
	#     config=config,
	#     stop=stop)