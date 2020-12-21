# Supression of Total Cross Sectional Scattering using Reinforcement Learning

## Description

<p>Using deep reinforcement learning to design a broadband acoustic cloak through inverse design</p>

<p>Example usage:</p>

```
from tscsRL.environments.ContinuousTSCSEnv import ContinuousTSCSEnv
from tscsRL.agents import ddpg

env = ContinuousTSCSEnv(
	nCyl=2,
	kMax=0.45,
	kMin=0.35,
	nFreq=11,
	stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 100
params['decay_timesteps'] = 100
params['num_episodes'] = 120
params['noise_scale'] = 1.1
params['save_data'] = False
params['use_wandb'] = True

name = 'example_run'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	env.stepSize, 
	params, 
	name)

agent.learn(env)	
```

<h3>DDPG diagram</h3>
<p align="center">
	<img src="https://github.com/gladisor/TSCSProject/blob/tristan/images/DDPG.png">
</p>

<h3>DDQN diagram</h3>
<p align="center">
	<img src="https://github.com/gladisor/TSCSProject/blob/tristan/images/DDQN.png">
</p>

## Credits
Images:
[Linwei Zhou](https://github.com/DiuLaMaX)

Inspiration for structuring agents:
[Ray](https://github.com/ray-project/ray)