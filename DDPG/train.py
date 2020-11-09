import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from env import TSCSEnv
import pandas as pd
from models import Actor, Critic
import wandb

def soft_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

def load_files():
    # Opening transition files
    tstate = pd.read_csv("state.csv", header=None).to_numpy()
    tactor = pd.read_csv("actor.csv", header=None).to_numpy()
    treward = pd.read_csv("reward.csv", header=None).to_numpy()
    tnextState = pd.read_csv("nextstate.csv", header=None).to_numpy()
    tdone = pd.read_csv("done.csv", header=None).to_numpy()

    # Convert files to torch tensors
    tstate = torch.tensor(tstate).float()
    tactor = torch.tensor(tactor).float()
    treward = torch.tensor(treward).float()
    tnextState = torch.tensor(tnextState).float()
    tdone = torch.tensor(tdone).float()

    # Indices for accessing dataset
    sizetransition = tstate.size()
    setlength = np.arange(0, sizetransition[0])

    return tstate,tactor,treward,tnextState,tdone,setlength

###############

wandb.init(project='tscs_train')

## env params
NCYL = 4
KMAX = 0.5
KMIN = 0.3
NFREQ = 11

# ddpg params
inSize = 21
actorNHidden = 2
actorHSize = 128
nActions = 2 * NCYL
actionRange = 0.2
criticNHidden = 8
criticHSize = 128
actorLR = 1e-4
criticLR = 1e-3
criticWD = 1e-2

gamma = 0.90  ## How much to value future reward
TAU = 0.001   ## How much to update target network every step

tstate,tactor,treward,tnextState,tdone,setlength = load_files()

## Define networks
actor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
targetActor = Actor(inSize, actorNHidden, actorHSize, nActions, actionRange)
critic = Critic(inSize, criticNHidden, criticHSize, nActions)
targetCritic = Critic(inSize, criticNHidden, criticHSize, nActions)

## Define the optimizers for both networks
actorOpt = Adam(actor.parameters(), lr=actorLR)
criticOpt = Adam(critic.parameters(), lr=criticLR, weight_decay=criticWD)

targetActor.load_state_dict(actor.state_dict())
targetCritic.load_state_dict(critic.state_dict())

env = TSCSEnv(NCYL, KMAX, KMIN, NFREQ)

batch_size = 200

batches = DataLoader(dataset=setlength, batch_size=batch_size, shuffle=True)

for i, batch in enumerate(tqdm(batches)):
    ## Get data from memory
    batch = batch.numpy()
    s = tstate[[batch]]
    a = tactor[[batch]]
    r = treward[[batch]]
    s_ = tnextState[[batch]]
    done = tdone[[batch]]

    env.reset()

    episode_reward = 0

    initial = env.RMS.item()
    lowest = initial

    for t in range(100):

        ## Compute target
        maxQ = targetCritic(s_[t], targetActor(s_[t]).detach())
        target_q = r[t] + (1.0 - done[t]) * gamma * maxQ

        ## Update the critic network
        criticOpt.zero_grad()
        current_q = critic(s[t], a[t])
        criticLoss = F.smooth_l1_loss(current_q, target_q.detach())
        criticLoss.backward()
        criticOpt.step()

        ## Update the actor network
        actorOpt.zero_grad()
        actorLoss = -critic(s[t], actor(s[t])).mean()
        actorLoss.backward()
        actorOpt.step()

        ## Copy policy weights over to target net
        soft_update(targetActor, actor)
        soft_update(targetCritic, critic)

        ## Evaluate
        episode_reward += torch.sum(r[t])

        # Get lowest RMS
        env.step(a[t])

        current = env.RMS.item()
        if current < lowest:
            lowest = current

        numIllegalMoves = env.numIllegalMoves

    wandb.log({
        'lowest': lowest,
        'score': episode_reward,
        'critic loss': criticLoss,
        'actor loss': actorLoss,
        'illegal moves': numIllegalMoves})
