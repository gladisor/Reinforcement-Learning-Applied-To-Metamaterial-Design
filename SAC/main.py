if __name__ == '__main__':
	from env import TSCSEnv
	import wandb
	import tqdm
	from sacGithub import SACAgent

	params = {
		'gamma': 0.90,
		'tau': 0.005,
		'alpha': 0.2,
		'batch_size': 256,
		'mem_size': 1_000_000,
		'num_random_episodes': 100,
		'save_every': 100,
		'run_name': 'firstRun',

		'env': TSCSEnv,
		'env_params': {
			'nCyl': 4,
			'nFreq': 11,
			'actionRange': 0.5,
			'episodeLength': 100,
			'k0amax': 0.45,
			'k0amin': 0.35},

		'alpha_lr': 3e-4,
		'actor_lr': 3e-4,
		'critic_lr': 3e-4,
		'actor_fc': [256] * 2,
		'critic_fc': [256] * 2,
	}

	sac = SACAgent(params)

	wandb.init(project='tscs-sac', config=params)

	for episode in range(20_000):
		## Initialize environment
		state = sac.env.reset()
		episodeReward = 0
		lowestMeanTSCS = sac.env.TSCS.mean()
		losses = {
			'critic':[],
			'actor':[],
			'alpha':[]}

		done = False
		while not done:
			## Select action
			if episode < params['num_random_episodes']:
				action = sac.env.action_space.sample()
			else:
				action = sac.get_action(state)

			## Apply action and store transition
			state_, reward, done = sac.env.step(action)
			sac.replay_buffer.push(state, action, reward, state_, done)
			state = state_

			## Do learning update
			loss = sac.update()

			## Collect metrics from update
			if loss != None:
				losses['critic'].append(loss[0])
				losses['actor'].append(loss[1])
				losses['alpha'].append(sac.log_alpha.exp().cpu().item())

			## Update episodeReward and lowest scattering found
			episodeReward += reward
			if sac.env.TSCS.mean() < lowestMeanTSCS:
				lowestMeanTSCS = sac.env.TSCS.mean()

		results = {
			'criticLoss': sum(losses['critic']),
			'actorLoss': sum(losses['actor']),
			'alpha': sum(losses['alpha']),
			'reward': episodeReward,
			'lowest': lowestMeanTSCS}

		## Save checkpoint
		if episode % params['save_every'] == 0:
			sac.save_models(params['run_name'])

		## Log and print results
		wandb.log(results)
		print(episode, results)

