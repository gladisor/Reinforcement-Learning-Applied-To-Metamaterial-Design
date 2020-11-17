import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from models import Actor, Critic
from memory import ReplayBuffer
from copy import deepcopy
import os

class SACAgent:
  
    def __init__(self, params):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.env = params['env'](params['env_params'])
        self.action_range = [self.env.action_space.low, self.env.action_space.high]
        self.obs_dim = self.env.observation_space.shape[1]
        self.action_dim = self.env.action_space.shape[0]

        # hyperparameters
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.update_step = 0
        self.delay_step = 1
        self.batch_size = params['batch_size']
        
        # initialize networks
        self.policy_net = Actor(
            lr=params['actor_lr'],
            inSize=self.obs_dim,
            fc=params['actor_fc'],
            nActions=self.action_dim,
            actionRange=params['env_params']['actionRange'])

        self.q_net1 = Critic(
            lr=params['critic_lr'],
            inSize=self.obs_dim,
            fc=params['critic_fc'],
            nActions=self.action_dim)

        self.q_net2 = Critic(
            lr=params['critic_lr'],
            inSize=self.obs_dim,
            fc=params['critic_fc'],
            nActions=self.action_dim)

        self.target_q_net1 = deepcopy(self.q_net1)
        self.target_q_net2 = deepcopy(self.q_net2)

        # entropy temperature
        self.alpha = params['alpha']
        self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=params['alpha_lr'])

        self.replay_buffer = ReplayBuffer(
            params['mem_size'], 
            state_shape=self.obs_dim, 
            action_shape=self.action_dim)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, log_std = self.policy_net.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().numpy()
        
        return action * self.action_range[1][0].item()

    def update(self):
        if not self.replay_buffer.can_provide_sample(self.batch_size):
            return

        ## Sampling transitions from memory
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        ## Converting to tensors
        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.long).to(self.device)

        ## Getting next actions and log_pi from next_states
        next_actions, next_log_pi = self.policy_net.sample(next_states)
        ## Scaling actions
        next_actions = (next_actions * self.action_range[1][0].item()).detach()

        ## Computing q target
        next_q1 = self.target_q_net1(next_states, next_actions)
        next_q2 = self.target_q_net2(next_states, next_actions)
        next_q_target = torch.min(next_q1, next_q2) - self.alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target.detach()

        ## q loss
        curr_q1 = self.q_net1.forward(states, actions)
        curr_q2 = self.q_net2.forward(states, actions)        
        q1_loss = F.smooth_l1_loss(curr_q1, expected_q.detach())
        q2_loss = F.smooth_l1_loss(curr_q2, expected_q.detach())

        ## update q networks        
        self.q_net1.opt.zero_grad()
        q1_loss.backward()
        self.q_net1.opt.step()
        
        self.q_net2.opt.zero_grad()
        q2_loss.backward()
        self.q_net2.opt.step()
        
        ## delayed update for policy network and target q networks
        ## Get new actions and log_pi from states
        new_actions, log_pi = self.policy_net.sample(states)
        ## Scaling actions
        new_actions = (new_actions * self.action_range[1][0].item()).detach()
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_net1.forward(states, new_actions),
                self.q_net2.forward(states, new_actions))
            
            policy_loss = (self.alpha * log_pi - min_q).mean()
            
            self.policy_net.opt.zero_grad()
            policy_loss.backward()
            self.policy_net.opt.step()
        
            # target networks
            self.soft_update(self.target_q_net1, self.q_net1)
            self.soft_update(self.target_q_net2, self.q_net2)

        # update temperature
        alpha_loss = (self.log_alpha * (-log_pi - self.target_entropy).detach()).mean()

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp()

        if self.update_step % self.delay_step == 0:
            self.update_step += 1
            return (q1_loss + q2_loss).item(), policy_loss.item(), alpha_loss.item()
        else:
            self.update_step += 1

    def save_models(self, name):
        print('---saving---')
        path = f'savedModels/{name}/'

        if not os.path.exists(path):
            os.mkdir(path)

        self.policy_net.save_checkpoint(path + 'policy_net.pt')
        self.target_q_net1.save_checkpoint(path + 'target_q_net1.pt')
        self.target_q_net2.save_checkpoint(path + 'target_q_net2.pt')

    def load_models(self, name):
        print('---loading---')
        path = f'savedModels/{name}/'
        self.policy_net.load_checkpoint(path + 'policy_net.pt')
        self.target_q_net1.load_checkpoint(path + 'target_q_net1.pt')
        self.target_q_net2.load_checkpoint(path + 'target_q_net2.pt') 