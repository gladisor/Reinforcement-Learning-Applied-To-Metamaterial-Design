from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy

class TSCSCallback(DefaultCallbacks):
	def on_episode_end(self, *, 
		worker: RolloutWorker, 
		base_env: BaseEnv,
		policies: Dict[str, Policy],
		episode: MultiAgentEpisode,
		env_index: int, **kwargs):