import gym
import ray
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
import pandas as pd
import matplotlib.pyplot as plt

ray.init()
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 10
config['num_gpus'] = 2
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config['num_cpus_per_worker'] = 0
config['framework'] = "torch"
agent = PPOTrainer(config, "CartPole-v0")

N = 10
episode_data = []
for n in range(N):
    result = agent.train()   
    episode = {
        'n': n,
        'episode_reward_mean': result['episode_reward_mean']}
    episode_data.append(episode)
    checkpoint = agent.save('my_checkpoint')
    print(f'episode: {n}, saved: {checkpoint}')

df = pd.DataFrame(data=episode_data)
print(df)