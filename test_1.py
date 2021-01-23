from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv, DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv, DiscreteGradientTSCSEnv
from tscsRL.agents import ddpg, ddqn
from tscsRL import utils
import imageio
import torch
import matlab

## Name of the run we want to evaluate
name = 'test_5'

path = 'results/' + name

env = ContinuousGradientTSCSEnv(
    nCyl=4,
    kMax=0.45,
    kMin=0.35,
    nFreq=11,
    stepSize=0.5)

params = ddpg.default_params()
params['save_every'] = 500
params['decay_timesteps'] = 8000
params['num_episodes'] = 5
params['noise_scale'] = 1.2
params['reward'] = []
params['lowest'] = []
params['save'] = False
params['plot_hpc'] = False
params['use_wandb'] = False


agent = ddpg.DDPGAgent(
    env.observation_space,
    env.action_space,
    params,
    name)

## Set exploration rate to low amount
agent.epsilon = 0.05
# agent.noise_scale = 0.02


## For creating a video of the episode
writer = imageio.get_writer(name + '.mp4', format='mp4', mode='I', fps=15)

## THIS WHOLE BLOCK IS THE INTERACTION LOOP

## Starting from a random config
state = env.reset()
## End starting from random config

# state = env.getState()
## End starting from random config
env.v = torch.zeros([1,8])

for i in range(500):
    state = env.getState()
    img = env.getIMG(env.config)
    #img = img.type(torch.uint8)
    writer.append_data(img.view(env.img_dim).numpy())

    action = agent.select_action(state)
    action = matlab.double(action.numpy().tolist())

    config = env.config
    config = matlab.double(config.numpy().tolist())

    velocity = env.v
    velocity = matlab.double(velocity.numpy().tolist())

    [q_, v_] = env.eng.step(action, config, velocity, matlab.int8([env.nCyl]), nargout=2)

    q_ = torch.FloatTensor(q_)
    v_ = torch.FloatTensor(v_)

    env.config = q_
    env.v = v_


writer.close()
