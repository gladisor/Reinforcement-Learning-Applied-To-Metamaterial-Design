from tscsRL.environments.TSCSEnv import ContinuousTSCSEnv
from tscsRL.environments.GradientTSCSEnv import ContinuousGradientTSCSEnv
from tscsRL.agents import ddpg
from tscsRL import utils

## Name of the run we want to evaluate
name = 'ddpg4cyl0.45-0.35-8000decay'

path = 'results/' + name
env_params = utils.jsonToDict(path + '/env_params.json')
agent_params = utils.jsonToDict(path + '/agent_params.json')

env = ContinuousTSCSEnv(
	nCyl=env_params['nCyl'],
	kMax=env_params['kMax'],
	kMin=env_params['kMin'],
	nFreq=env_params['nFreq'],
	stepSize=env_params['stepSize'])

agent = ddpg.DDPGAgent(
	env.observation_space, 
	env.action_space, 
	agent_params, 
	name + 'SecondRun')

agent.learn(env)