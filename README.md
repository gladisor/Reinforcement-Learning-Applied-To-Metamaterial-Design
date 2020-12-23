# Supression of Total Scattering Cross Section using Reinforcement Learning

<p>Our aim in this research is to use reinforcement learning to design a broadband acoustic cloak through inverse design</p>

<h3>Optimal control</h3>
<p align="center">
	<img src="https://github.com/gladisor/TSCSProject/blob/tristan/images/ddpg4cyl0.45-0.35-8000decay.gif">
</p>

<h3>Example usage:</h3>

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
params['action_range'] = env.stepSize
params['save_data'] = False
params['use_wandb'] = True

name = 'test_ddpg'

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	params, 
	name)

agent.learn(env)	
```

<h3>Diagrams of training loops</h3>
<p align="center">
	<img src="https://github.com/gladisor/TSCSProject/blob/tristan/images/DDPG.png" width="400">
	<img src="https://github.com/gladisor/TSCSProject/blob/tristan/images/DDQN.png" width="400">
</p>

## Credits
Images:
[Linwei Zhou](https://github.com/DiuLaMaX)

Inspiration for structuring agents:
[Ray](https://github.com/ray-project/ray)