'''DLP DDPG Lab'''
__author__ = 'chengscott'
__copyright__ = 'Copyright 2020, NCTU CGI Lab'
import argparse
from collections import deque
import itertools
import random
import time

import gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt


class GaussianNoise:
    def __init__(self, dim, mu=None, std=None):
        self.mu = mu if mu else np.zeros(dim)
        self.std = std if std else np.ones(dim) * .1

    def sample(self):
        return np.random.normal(self.mu, self.std)


class ReplayMemory:
    __slots__ = ['buffer']

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, *transition):
        # (state, action, reward, next_state, done)
        self.buffer.append(tuple(map(tuple, transition)))

    def sample(self, batch_size, device):
        '''sample a batch of transition tensors'''
        ## TODO ##
        #raise NotImplementedError
        transitions = random.sample(self.buffer, batch_size)
        return (torch.tensor(x, dtype=torch.float, device=device)
                for x in zip(*transitions))


class ActorNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        ## TODO ##
        #raise NotImplementedError
        self.layer1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim[0]),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU()
        )
        self.final = nn.Sequential(
            nn.Linear(hidden_dim[1], action_dim),
            nn.Tanh()
        )
        """
        self.fc1=nn.Linear(state_dim,hidden_dim[0])
        self.fc2=nn.Linear(hidden_dim[0],hidden_dim[1])
        self.fc3=nn.Linear(hidden_dim[1],action_dim)
        self.relu=nn.ReLU()
        self.tanh=nn.Tanh()
        """


    def forward(self, x):
        ## TODO ##
        #raise NotImplementedError
        
        x = self.layer1(x)
        x = self.layer2(x)
        output = self.final(x)
        return output
        """
        out=self.relu(self.fc1(x))
        out=self.relu(self.fc2(out))
        out=self.tanh(self.fc3(out))
        return out
        """

class CriticNet(nn.Module):
    def __init__(self, state_dim=8, action_dim=2, hidden_dim=(400, 300)):
        super().__init__()
        h1, h2 = hidden_dim
        self.critic_head = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
        )
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x, action):
        x = self.critic_head(torch.cat([x, action], dim=1))
        return self.critic(x)


class DDPG:
    def __init__(self, args):
        # behavior network
        self._actor_net = ActorNet().to(args.device)
        self._critic_net = CriticNet().to(args.device)
        # target network
        self._target_actor_net = ActorNet().to(args.device)
        self._target_critic_net = CriticNet().to(args.device)
        # initialize target network
        self._target_actor_net.load_state_dict(self._actor_net.state_dict())
        self._target_critic_net.load_state_dict(self._critic_net.state_dict())
        ## TODO ##
        self._actor_opt = torch.optim.Adam(self._actor_net.parameters(), lr=args.lra)
        self._critic_opt = torch.optim.Adam(self._critic_net.parameters(), lr=args.lrc)
        #raise NotImplementedError
        # action noise
        self._action_noise = GaussianNoise(dim=2)
        # memory
        self._memory = ReplayMemory(capacity=args.capacity)

        ## config ##
        self.device = args.device
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.gamma = args.gamma

    def select_action(self, state, noise=True):
        '''based on the behavior (actor) network and exploration noise'''
        ## TODO ##
        #raise NotImplementedError
        state = torch.as_tensor(state, device=self.device)
        #mu = self._actor_net(state).argmax().item()
        mu = self._actor_net(state)
        if noise:
            noise = torch.as_tensor(self._action_noise.sample(), device=self.device)
            return (mu + noise).tolist()
        else:
            return mu.tolist()

    def append(self, state, action, reward, next_state, done):
        self._memory.append(state, action, [reward / 100], next_state,
                            [int(done)])

    def update(self):
        loss = torch.zeros(2, device=self.device)
        # update the behavior networks
        loss = self._update_behavior_network(self.gamma)
        # update the target networks
        self._update_target_network(self._target_actor_net, self._actor_net,
                                    self.tau)
        self._update_target_network(self._target_critic_net, self._critic_net,
                                    self.tau)
        return loss

    def _update_behavior_network(self, gamma):
        actor_net, critic_net, target_actor_net, target_critic_net = self._actor_net, self._critic_net, self._target_actor_net, self._target_critic_net
        actor_opt, critic_opt = self._actor_opt, self._critic_opt

        # sample a minibatch of transitions
        state, action, reward, next_state, done = self._memory.sample(
            self.batch_size, self.device)

        ## update critic ##
        # critic loss
        ## TODO ##
        q_value = critic_net(state, action)
        with torch.no_grad():
            a_next = target_actor_net(next_state)
            q_next = target_critic_net(next_state, a_next)
            q_target = reward + (gamma * q_next) * (1 - done)
        criterion = nn.MSELoss()
        critic_loss = criterion(q_value, q_target)
        #raise NotImplementedError
        # optimize critic
        actor_net.zero_grad()
        critic_net.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        ## update actor ##
        # actor loss
        ## TODO ##
        action = actor_net(state)
        actor_loss = -critic_net(state, action).mean()
        #raise NotImplementedError
        # optimize actor
        actor_net.zero_grad()
        critic_net.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        
        loss_a = actor_loss.view(1)
        loss_c = critic_loss.view(1)
        return torch.cat([loss_a, loss_c])

    @staticmethod
    def _update_target_network(target_net, net, tau):
        '''update target network by _soft_ copying from behavior network'''
        for target, behavior in zip(target_net.parameters(), net.parameters()):
            ## TODO ##
            #raise NotImplementedError
            target.data.copy_(tau * behavior.data + (1 - tau) * target.data)


    def save(self, model_path, checkpoint=False):
        if checkpoint:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                    'target_actor': self._target_actor_net.state_dict(),
                    'target_critic': self._target_critic_net.state_dict(),
                    'actor_opt': self._actor_opt.state_dict(),
                    'critic_opt': self._critic_opt.state_dict(),
                }, model_path)
        else:
            torch.save(
                {
                    'actor': self._actor_net.state_dict(),
                    'critic': self._critic_net.state_dict(),
                }, model_path)

    def load(self, model_path, checkpoint=False):
        model = torch.load(model_path)
        self._actor_net.load_state_dict(model['actor'])
        self._critic_net.load_state_dict(model['critic'])
        if checkpoint:
            self._target_actor_net.load_state_dict(model['target_actor'])
            self._target_critic_net.load_state_dict(model['target_critic'])
            self._actor_opt.load_state_dict(model['actor_opt'])
            self._critic_opt.load_state_dict(model['critic_opt'])


def train(args, env, agent, writer):
    print('Start Training')
    total_steps = 0
    ewma_reward = 0
    rewards = []
    avg50_rewards = []
    losses = torch.zeros(args.episode, 2, device=args.device)
    avg_r = 0
    for episode in range(args.episode):
        total_reward = 0
        state = env.reset()
        loss = torch.zeros(2, device=args.device)
        for t in itertools.count(start=1):
            # select action
            if total_steps < args.warmup:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)
            # execute action
            next_state, reward, done, _ = env.step(action)
            # store transition
            agent.append(state, action, reward, next_state, done)
            if total_steps >= args.warmup:
                loss += agent.update()

            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Train/Episode Reward', total_reward,
                                  total_steps)
                writer.add_scalar('Train/Ewma Reward', ewma_reward,
                                  total_steps)
                print(
                    'Step: {}\tEpisode: {}\tLength: {:3d}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(total_steps, episode, t, total_reward,
                            ewma_reward))
                rewards.append(total_reward)
                avg_r += total_reward
                if episode % 50 == 0:
                    avg50_rewards.append(avg_r / 50)
                    avg_r = 0
                
                break
                
        losses[episode] = loss
    
    env.close()
    
    plt.title("Rewards")
    plt.plot(range(args.episode), rewards)
    plt.plot(range(0, args.episode, 50) , avg50_rewards, label="avg50")
    plt.show()
    plt.title("Loss")
    losses = losses.cpu().detach()
    plt.plot(range(args.episode), losses[:, 0], label="actor_loss")
    plt.plot(range(args.episode), losses[:, 1], label="critic_loss")
    plt.legend()
    plt.show()


def test(args, env, agent, writer):
    print('Start Testing')
    seeds = (args.seed + i for i in range(10))
    rewards = []
    ewma_reward = 0
    for n_episode, seed in enumerate(seeds):
        total_reward = 0
        env.seed(seed)
        state = env.reset()
        ## TODO ##
        for t in itertools.count(start=1):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            if done:
                ewma_reward = 0.05 * total_reward + (1 - 0.05) * ewma_reward
                writer.add_scalar('Test/Episode Reward', total_reward, n_episode)
                writer.add_scalar('Train/Ewma Reward', ewma_reward, n_episode)
                print(
                    'Episode: {}\tTotal reward: {:.2f}\tEwma reward: {:.2f}'
                    .format(n_episode, total_reward, ewma_reward))
                rewards.append(total_reward)
                break
        #raise NotImplementedError
    print('Average Reward', np.mean(rewards))
    env.close()

## arguments ##
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('-d', '--device', default='cuda:2')
parser.add_argument('-m', '--model', default='ddpg_final_1800.pth')
parser.add_argument('--logdir', default='log/ddpg')
# train
parser.add_argument('--warmup', default=10000, type=int)
parser.add_argument('--episode', default=1800, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--capacity', default=500000, type=int)
parser.add_argument('--lra', default=1e-3, type=float)
parser.add_argument('--lrc', default=1e-3, type=float)
parser.add_argument('--gamma', default=.99, type=float)
parser.add_argument('--tau', default=.005, type=float)
# test
parser.add_argument('--test_only', action='store_true')
parser.add_argument('--render', action='store_true')
parser.add_argument('--seed', default=20200519, type=int)
args = parser.parse_args(args=[])

path = "/home/auser03/lab6/models/"

## main ##
env = gym.make('LunarLanderContinuous-v2')
agent = DDPG(args)
writer = SummaryWriter(args.logdir)

'''
    train
'''
if not args.test_only:
    train(args, env, agent, writer)
    agent.save(path+args.model)

'''
    test
'''
agent.load(path+args.model)
test(args, env, agent, writer)
