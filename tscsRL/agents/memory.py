import numpy as np

class NaivePrioritizedBuffer():
	def __init__(self, capacity, alpha):
		self.alpha = alpha
		self.capacity = capacity
		self.memory = []
		self.idx = 0
		self.priorities = np.ones((capacity,), dtype=np.float32)

	def push(self, transition):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.idx] = transition
		self.priorities[self.idx] = self.priorities.max()
		self.idx = (self.idx + 1) % self.capacity

	def sample(self, batch_size, beta):
		if len(self.memory) == self.capacity:
			prios = self.priorities
		else:
			prios = self.priorities[:self.idx]

		probs = prios ** self.alpha
		probs /= probs.sum()

		indices = np.random.choice(len(self.memory), batch_size, p=probs, replace=False)
		samples = [self.memory[idx] for idx in indices]

		total = len(self.memory)
		weights = (total * probs[indices]) ** (-beta)
		weights /= weights.max()
		weights = np.array(weights, dtype=np.float32)
		return samples, indices, weights

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

	def update_priorities(self, batch_indices, batch_priorities):
		for idx, prio in zip(batch_indices, batch_priorities):
			self.priorities[idx] = prio

	def __len__(self):
		return len(self.memory)