import json
import wandb
import matplotlib.pyplot as plt

class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__


def init_wandb(params):
    wandb.init(project=params.PROJECT_NAME)
    wandb.config.nCyl = params.NCYL
    wandb.config.kmax = params.KMAX
    wandb.config.kmin = params.KMIN
    wandb.config.nfreq = params.NFREQ
    wandb.config.actor_n_hidden = params.ACTOR_N_HIDDEN
    wandb.config.actor_h_size = params.ACTOR_H_SIZE
    wandb.config.critic_n_hidden = params.CRITIC_N_HIDDEN
    wandb.config.critic_h_size = params.CRITIC_H_SIZE
    wandb.config.action_range = params.ACTION_RANGE
    wandb.config.actor_lr = params.ACTOR_LR
    wandb.config.critic_lr = params.CRITIC_LR
    wandb.config.critic_wd = params.CRITIC_WD
    wandb.config.gamma = params.GAMMA
    wandb.config.tau = params.TAU
    wandb.config.epsilon = params.EPSILON
    wandb.config.eps_decay = params.EPS_DECAY
    wandb.config.eps_end = params.EPS_END
    wandb.config.mem_size = params.MEM_SIZE
    wandb.config.alpha = params.MEM_ALPHA
    wandb.config.beta = params.MEM_BETA
    wandb.config.batch_size = params.BATCH_SIZE


def log_wandb(epsilon, lowest, episode_reward):
    wandb.log({
        'epsilon': epsilon,
        'lowest': lowest,
        'score': episode_reward})

def plot(name, data):
    plt.figure()
    plt.plot(data, 'r')
    plt.title(name)
    plt.xlabel('Epochs')
    plt.ylabel(name)
    plt.savefig('result/figures/' + name)