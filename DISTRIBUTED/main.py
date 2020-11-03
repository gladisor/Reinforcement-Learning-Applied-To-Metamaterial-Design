import ray
from ray.rllib.agents.ddpg import apex
from ray.rllib.agents import impala
from ray import tune
from env import DistributedTSCSEnv

if __name__ == '__main__':
	config = impala.DEFAULT_CONFIG.copy()
	config['rollout_fragment_length'] = 100
	config['train_batch_size'] = 500
	config['num_workers'] = 8
	config['num_envs_per_worker'] = 2
	config['num_gpus'] = 1
	config['framework'] = 'torch'
	config['gamma'] = 0.90
	config['env'] = DistributedTSCSEnv
	config['env_config'] = {
	    'nCyl': 4,
	    'k0amax': 0.45,
	    'k0amin': 0.35,
	    'nFreq': 11,
	    'actionRange': 0.5,
	    'episodeLength': 100}

	tune.run(
	    impala.ImpalaTrainer,
	    stop={'episode_reward_mean': 0},
	    config=config)