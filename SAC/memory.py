import numpy as np

class ReplayBuffer():
	"""docstring for ReplayBuffer"""
	def __init__(self, capacity, state_shape, action_shape):
		self.capacity = capacity
		self.mem_ctr = 0
		self.memory = {
			's': np.zeros((self.capacity, state_shape)),
			'a': np.zeros((self.capacity, action_shape)),
			'r': np.zeros((self.capacity, 1)),
			's_': np.zeros((self.capacity, state_shape)),
			'done': np.zeros((self.capacity, 1))}

	def can_provide_sample(self, batch_size):
		return self.mem_ctr >= batch_size

	def push(self, s, a, r, s_, done):
		idx = self.mem_ctr % self.capacity
		self.memory['s'][idx] = s
		self.memory['a'][idx] = a
		self.memory['r'][idx] = r
		self.memory['s_'][idx] = s_
		self.memory['done'][idx] = done

		self.mem_ctr += 1

	def sample(self, batch_size):
		max_mem = min(self.mem_ctr, self.capacity)
		batch = np.random.choice(a=max_mem, size=batch_size, replace=False)

		s = self.memory['s'][batch]
		a = self.memory['a'][batch]
		r = self.memory['r'][batch]
		s_ = self.memory['s_'][batch]
		done = self.memory['done'][batch]

		return s, a, r, s_, done

if __name__ == '__main__':
	from env import TSCSEnv


	params = {
		'nCyl': 4,
		'nFreq': 11,
		'actionRange': 0.5,
		'episodeLength': 100,
		'k0amax': 0.45,
		'k0amin': 0.35}

	env = TSCSEnv(params)

	memory = ReplayBuffer(
		100, 
		state_shape=env.observation_space.shape[1], 
		action_shape=env.action_space.shape[0])

	state = env.reset()

	done = False
	while not done:
		action = env.action_space.sample()
		state_, reward, done = env.step(action)

		memory.push(state, action, reward, state_, done)
		print(done)
		state = state_

	s, a, r, s_, done = memory.sample(100)
	print(done)

