from models import ImageActor, ImageCritic
import torch
from torch import tensor, cat, tanh
from torch.optim import Adam
import torch.nn.functional as F
from collections import namedtuple
from memory import NaivePrioritizedBuffer
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from new_env import TSCSEnv
import wandb
from noise import OrnsteinUhlenbeckActionNoise
import utils


class DDPG():
    def __init__(self, params):

        super(DDPG, self).__init__()
        ## Actions
        self.nActions = params.N_ACTIONS
        self.actionRange = params.ACTION_RANGE

        ## Networks
        self.actor = ImageActor(params.IN_SIZE, params.ACTOR_N_HIDDEN, params.ACTOR_H_SIZE, params.N_ACTIONS, params.ACTION_RANGE)
        self.targetActor = ImageActor(params.IN_SIZE, params.ACTOR_N_HIDDEN, params.ACTOR_H_SIZE, params.N_ACTIONS, params.ACTION_RANGE)
        self.critic = ImageCritic(params.IN_SIZE, params.CRITIC_N_HIDDEN, params.CRITIC_H_SIZE, params.N_ACTIONS)
        self.targetCritic = ImageCritic(params.IN_SIZE, params.CRITIC_N_HIDDEN, params.CRITIC_H_SIZE, params.N_ACTIONS)

        if torch.cuda.is_available():
            self.actor, self.targetActor = self.actor.cuda(), self.targetActor.cuda()
            self.critic, self.targetCritic = self.critic.cuda(), self.targetCritic.cuda()

        ## Define the optimizers for both networks
        self.actorOpt = Adam(self.actor.parameters(), lr=params.ACTOR_LR)
        self.criticOpt = Adam(self.critic.parameters(), lr=params.CRITIC_LR, weight_decay=params.CRITIC_WD)

        ## Hard update
        self.targetActor.load_state_dict(self.actor.state_dict())
        self.targetCritic.load_state_dict(self.critic.state_dict())

        ## Various hyperparameters
        self.gamma = params.GAMMA
        self.tau = params.TAU
        self.epsilon = params.EPSILON
        self.epsStart = params.EPSILON
        self.epsDecay = params.EPS_DECAY
        self.epsEnd = params.EPS_END

        ## Transition tuple to store experience
        self.Transition = namedtuple(
            'Transition',
            ('s','img', 'a', 'r', 's_', 'nextImg','done'))

        ## Allocate memory for replay buffer and set batch size
        self.memory = NaivePrioritizedBuffer(params.MEM_SIZE)
        self.batchSize = params.BATCH_SIZE

        self.numEpisodes = params.NUM_EPISODES
        self.epLen = params.EP_LEN
        self.saveModels = params.saveModels

    def select_action(self, img, state):
        with torch.no_grad():
            noise = np.random.normal(0, self.epsilon, self.nActions)
            action = self.targetActor(img.cuda(), state.cuda()).cpu() + torch.tensor([noise])
            action.clamp_(-self.actionRange, self.actionRange)
        return action

    def select_action_ou(self, img, state):
        with torch.no_grad():
            action = self.targetActor(img.cuda(), state.cuda()).cpu() + self.noise()
            action.clamp_(-self.actionRange, self.actionRange)
        return action

    def extract_tensors(self, batch):
        batch = self.Transition(*zip(*batch))
        s = cat(batch.s)
        img = cat(batch.img)
        a = cat(batch.a)
        r = cat(batch.r)
        s_ = cat(batch.s_)
        img_ = cat(batch.nextImg)
        done = cat(batch.done)
        return s, img, a, r, s_, img_, done

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def optimize_model(self):
        if self.memory.can_provide_sample(self.batchSize):
            ## Get data from memory
            batch, indices, weights = self.memory.sample(self.batchSize)
            s, img, a, r, s_, img_, done = self.extract_tensors(batch)
            s, img, a, r, s_, img_, done = s.cuda(), img.cuda(), a.cuda(), r.cuda(), s_.cuda(), img_.cuda(), done.cuda()
            weights = tensor([weights]).cuda()

            ## Compute target
            maxQ = self.targetCritic(img_, s_, self.targetActor(img_, s_).detach())
            target_q = r + (1.0 - done) * self.gamma * maxQ

            ## Update the critic network
            self.criticOpt.zero_grad()
            current_q = self.critic(img,s, a)
            criticLoss = weights @ F.smooth_l1_loss(current_q, target_q.detach(), reduction='none')
            criticLoss.backward()
            self.criticOpt.step()

            ## Update the actor network
            self.actorOpt.zero_grad()
            actorLoss = -self.critic(img, s, self.actor(img, s)).mean()
            actorLoss.backward()
            self.actorOpt.step()

            ## Copy policy weights over to target net
            self.soft_update(self.targetActor, self.actor)
            self.soft_update(self.targetCritic, self.critic)

            ## Updating priority of transition by last absolute td error
            td = torch.abs(target_q - current_q).detach()
            self.memory.update_priorities(indices, td + 1e-5)
            return td.mean().item()

    def decay_epsilon(self):
        self.epsilon -= (self.epsStart - self.epsEnd) / self.epsDecay
        self.epsilon = max(self.epsilon, self.epsEnd)

    def evaluate(self, env):
        state = env.reset()
        episode_reward = 0

        initial = env.RMS.item()
        lowest = initial

        for t in tqdm(range(self.epLen), desc="eval"):
            with torch.no_grad():
                action = self.targetActor(state.cuda()).cpu()

            nextState, reward = env.step(action)
            episode_reward += reward

            current = env.RMS.item()
            if current < lowest:
                lowest = current

            state = nextState
        return episode_reward, lowest

    def learn(self, env, params):
        ## Create file to store run data in using tensorboard

        for episode in range(self.numEpisodes):

            ## Reset environment to starting state
            state = env.reset()

            episode_reward = 0

            ## Log initial scattering at beginning of episode
            initial = env.RMS.item()
            lowest = initial
            params.lowest = initial
            print('episode: ' + str(episode) + '\n')
            for t in tqdm(range(self.epLen), desc="train"):

                ## Select action and observe next state, reward
                img = env.img
                action = self.select_action(img, state)
                nextState, reward = env.step(action)
                nextImage = env.img
                episode_reward += reward

                # Update current lowest scatter
                current = env.RMS.item()

                if current < lowest:
                    lowest = current 


                ## Check if terminal
                if t == self.epLen - 1:
                    done = 1
                else:
                    done = 0

                ## Cast reward and done as tensors
                reward = tensor([[reward]]).float()
                done = tensor([[done]])

                ## Store transition in memory
                self.memory.push(self.Transition(state, img, action, reward, nextState, nextImage, done))

                ## Preform bellman update
                td = self.optimize_model()

                ## Break out of loop if terminal state
                if done == 1:
                    break

                state = nextState


            ## Print episode statistics to console
            print('lowest: ' + str(lowest) + '\n')
            print('episode_reward: ' + str(episode_reward) + '\n')
            print('Invalid Move: ' + str(env.numIllegalMoves) + '\n')
            print('epsilon: ' + str(self.epsilon) + '\n')

            params.epsilon = np.append(params.epsilon, self.epsilon)
            params.reward = np.append(params.reward, episode_reward)
            params.lowest = np.append(params.lowest, lowest)
            # Update result to wandb
            # utils.logWandb(self.epsilon, lowest, episode_reward)


            ## Save models
            if episode % (self.saveModels - 1) == 0:
                print("Saving Model")
                actorCheckpoint = {'state_dict': self.actor.state_dict()}
                targetActorCheckpoint = {'state_dict': self.targetActor.state_dict()}
                criticCheckpoint = {'state_dict': self.critic.state_dict()}
                targetCriticCheckpoint = {'state_dict': self.targetCritic.state_dict()}
                torch.save(actorCheckpoint, 'savedModels/actor.pth.tar')
                torch.save(targetActorCheckpoint, 'savedModels/targetActor.pth.tar')
                torch.save(criticCheckpoint, 'savedModels/critic.pth.tar')
                torch.save(targetCriticCheckpoint, 'savedModels/targetCritic.pth.tar')

            ## Reduce exploration
            self.decay_epsilon()


if __name__ == '__main__':

    params = utils.Params('Params.json')
    params.N_ACTIONS = int(2 * params.NCYL)
    params.reward = np.array([])
    params.lowest = np.array([])
    params.epsilon = np.array([]) 
    agent = DDPG(params)
     
    # Setting memory hyperparameters
    agent.memory.alpha = params.MEM_ALPHA
    agent.memory.beta = params.MEM_BETA

    # utils.initWandb(params)
    actorCheckpoint = torch.load('savedModels/actor.pth.tar')
    criticCheckpoint = torch.load('savedModels/critic.pth.tar')
    targetActorCheckpoint = torch.load('savedModels/targetActor.pth.tar')
    targetCriticCheckpoint= torch.load('savedModels/targetCritic.pth.tar') 
    agent.actor.load_state_dict(actorCheckpoint['state_dict'])
    agent.critic.load_state_dict(criticCheckpoint['state_dict'])
    agent.targetActor.load_state_dict(targetActorCheckpoint['state_dict'])
    agent.targetCritic.load_state_dict(targetCriticCheckpoint['state_dict'])
    
    # Create env and agent
    env = TSCSEnv(params)

    # Run training session
    agent.learn(env, params)

    # plot and save data
    numIter = 300
    utils.plot('reward' + str(numIter), params.reward)
    utils.plot('lowest' + str(numIter), params.lowest)
    utils.plot('epsilon' + str(numIter), params.epsilon)
    maxReward = max(params.reward)
    minTSCS = min(params.lowest)
    minEpsilon = min(params.epsilon)
    result = {'maxReward' + str(numIter): maxReward,
            'minTSCS' + str(numIter): minTSCS,
            'minEpsilon' + str(numIter): minEpsilon
            }
    utils.saveData(result)

