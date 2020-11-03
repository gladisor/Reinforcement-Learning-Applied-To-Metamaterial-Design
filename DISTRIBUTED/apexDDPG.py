import ray
from ray.rllib.agents.ddpg import apex
from ray import tune
from env import DistributedTSCSEnv

config = {
	'num_workers': 8,
	'num_envs_per_worker': 1,
	'rollout_fragment_length': 100,
	'batch_mode': 'truncate_episodes',
	'num_gpus': 1,
	'train_batch_size': 512, 
	'model': 
		{
			'fcnet_hiddens': [256, 256],
			'fcnet_activation': 'tanh', 
			'conv_filters': None, 
			'conv_activation': 'relu', 
			'free_log_std': False, 
			'no_final_linear': False, 
			'vf_share_layers': True, 
			'use_lstm': False, 
			'max_seq_len': 20, 
			'lstm_cell_size': 256, 
			'lstm_use_prev_action_reward': False, 
			'_time_major': False, 
			'framestack': True, 
			'dim': 84, 
			'grayscale': False, 
			'zero_mean': True, 
			'custom_model': None, 
			'custom_model_config': {}, 
			'custom_action_dist': None, 
			'custom_preprocessor': None
		},
	'optimizer': 
		{
			'max_weight_sync_delay': 400,
			'num_replay_buffer_shards': 4, 
			'debug': False
		},
	'gamma': 0.90,
	'horizon': None,
	'soft_horizon': False,
	'no_done_at_end': False, 
	'env_config': 
		{
			'nCyl': 4,
	        'k0amax': 0.45,
	        'k0amin': 0.35,
	        'nFreq': 11,
	        'actionRange': 0.5,
	        'episodeLength': 100
		}, 
	'env': DistributedTSCSEnv, 
	'normalize_actions': False, 
	'clip_rewards': None, 
	'clip_actions': True,
	'preprocessor_pref': 'deepmind', 
	'lr': 0.0001,
	'monitor': False,
	'log_level': 'WARN',
	'callbacks': ray.rllib.agents.callbacks.DefaultCallbacks,
	'ignore_worker_failures': False, 
	'log_sys_usage': True, 
	'fake_sampler': False, 
	'framework': 'torch',
	'eager_tracing': False,
	'no_eager_on_workers': False,
	'explore': True,
	'exploration_config': 
		{
			'type': 'PerWorkerOrnsteinUhlenbeckNoise'
		},
	'evaluation_interval': None, 
	'evaluation_num_episodes': 10, 
	'in_evaluation': False, 
	'evaluation_config': 
		{
			'explore': False
		},
	'evaluation_num_workers': 0,
	'custom_eval_function': None, 
	'sample_async': False, 
	'_use_trajectory_view_api': False, 
	'observation_filter': 'NoFilter', 
	'synchronize_filters': True, 
	'compress_observations': False, 
	'collect_metrics_timeout': 180, 
	'metrics_smoothing_episodes': 100, 
	'remote_worker_envs': False, 
	'remote_env_batch_wait_ms': 0, 
	'min_iter_time_s': 30, 
	'timesteps_per_iteration': 25000, 
	'seed': None, 
	'extra_python_environs_for_driver': {}, 
	'extra_python_environs_for_worker': {}, 
	'num_cpus_per_worker': 1, 
	'num_gpus_per_worker': 0, 
	'custom_resources_per_worker': {}, 
	'num_cpus_for_driver': 1, 
	'memory': 0, 
	'object_store_memory': 0, 
	'memory_per_worker': 0, 
	'object_store_memory_per_worker': 0, 
	'input': 'sampler', 
	'input_evaluation': ['is', 'wis'], 
	'postprocess_inputs': False, 
	'shuffle_buffer_size': 0, 
	'output': None, 
	'output_compress_columns': ['obs', 'new_obs'], 
	'output_max_file_size': 67108864, 
	'multiagent': 
		{
			'policies': {}, 
			'policy_mapping_fn': None, 
			'policies_to_train': None, 
			'observation_fn': None, 
			'replay_mode': 'independent'
		}, 
	'logger_config': None, 
	'replay_sequence_length': 1, 
	'twin_q': False, 
	'policy_delay': 1, 
	'smooth_target_policy': False, 
	'target_noise': 0.2, 
	'target_noise_clip': 0.5, 
	'use_state_preprocessor': False, 
	'actor_hiddens': [400, 300], 
	'actor_hidden_activation': 'relu', 
	'critic_hiddens': [400, 300], 
	'critic_hidden_activation': 'relu', 
	'n_step': 3, 
	'buffer_size': 2000000, 
	'prioritized_replay': True, 
	'prioritized_replay_alpha': 0.6, 
	'prioritized_replay_beta': 0.4, 
	'prioritized_replay_beta_annealing_timesteps': 20000, 
	'final_prioritized_replay_beta': 0.4,
	'prioritized_replay_eps': 1e-06, 
	'training_intensity': None, 
	'critic_lr': 0.001, 
	'actor_lr': 0.001, 
	'target_network_update_freq': 500000, 
	'tau': 0.002, 
	'use_huber': False, 
	'huber_threshold': 1.0, 
	'l2_reg': 1e-06, 
	'grad_clip': None, 
	'learning_starts': 50000, 
	'worker_side_prioritization': True
}

agent = apex.ApexDDPGTrainer(config)

# tune.run(
# 	apex.ApexDDPGTrainer,
# 	stop={'episode_reward_mean':-30},
# 	config=config)