from tscsRL.environments.TSCSEnv import DiscreteTSCSEnv
from tscsRL.environments.GradientTSCSEnv import DiscreteGradientTSCSEnv
from tscsRL.agents import ddqn
from tscsRL import utils

## Name of the run we want to evaluate
name = 'ddqn4cyl0.45-0.35-8000decay'

path = 'results/' + name
env_params = utils.jsonToDict(path + '/env_params.json')
agent_params = utils.jsonToDict(path + '/agent_params.json')

env = DiscreteTSCSEnv(
	nCyl=env_params['nCyl'],
	kMax=env_params['kMax'],
	kMin=env_params['kMin'],
	nFreq=env_params['nFreq'],
	stepSize=env_params['stepSize'])

agent = ddqn.DDQNAgent(
	env.observation_space, 
	env.action_space,
	agent_params, 
	name + 'SecondRun')

agent.learn(env)