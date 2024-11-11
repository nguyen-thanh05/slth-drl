"""
VPG without baseline (REINFORCE)?


"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque, namedtuple
# import categorical
from torch.distributions import Categorical


from types import SimpleNamespace

import gymnasium as gym
from Layers import SupermaskConv2d, SupermaskLinear




class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, sparsity):
        super(Actor, self).__init__()
        self.fc1 = SupermaskLinear(sparsity=sparsity, in_features=state_dim, out_features=128)
        self.fc2 = SupermaskLinear(sparsity=sparsity, in_features=128, out_features=action_dim)
        return

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x

class Trajectory_buffer:
    def __init__(self, max_size):
        self.obs_buffer = [None] * max_size
        self.rew_buffer = [None] * max_size
        self.act_buffer = [None] * max_size
        self.log_prob_buffer = [None] * max_size
        self.ptr = 0
        self.max_size = max_size
        return

    def get(self):
        return self.obs_buffer, self.act_buffer, self.rew_buffer, self.log_prob_buffer

    def add(self, obs, act, rew, log_prob):
        self.obs_buffer[self.ptr] = obs
        self.act_buffer[self.ptr] = act
        self.rew_buffer[self.ptr] = rew
        self.log_prob_buffer[self.ptr] = log_prob

        self.ptr = (self.ptr + 1) % self.max_size

    def reset(self):
        self.ptr = 0
        return


class VPG:
    def __init__(self, config: SimpleNamespace):
        self.env = gym.make(config.env)
        self.batch = config.batch
        self.seed = config.seed
        self.num_epochs = config.num_epochs
        self.steps_per_epoch = config.steps_per_epoch   # length of the trajectory per epoch
        self.sparsity = config.sparsity
        self.gamma = 0.99
        self.device = config.device

        self.actor = Actor(self.env.observation_space.shape[0], self.env.action_space.n, sparsity=self.sparsity)
        self.trajectory_buffer = Trajectory_buffer(self.steps_per_epoch)

        self.optimizer = torch.optim.Adam(self.actor.parameters())
        return

    def collect(self):
        state, info = self.env.reset()
        print("state ", state)
        for i in range(self.steps_per_epoch):
            probs = self.actor(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))
            m = Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            v = 0

            next_state, reward, truncated, done, info = self.env.step(action.item())
            self.trajectory_buffer.add(state, action, reward, log_prob)
            state = next_state

            if done or truncated:
                state = self.env.reset()
        return

    def update(self):
        
        return

    def train_VPG(self):
        for epoch in range(self.num_epochs):
            self.collect()

            obs_buf, act_buf, rew_buf, log_prob_buf = self.trajectory_buffer.get()

            R_t_buff = [0] * self.steps_per_epoch
            for i in range(self.steps_per_epoch):  # compute R_t
                R_t_buff[i] = sum([self.gamma**t * rew_buf[t] for t in range(i, self.steps_per_epoch)])

            R_t = torch.tensor(R_t_buff, dtype=torch.float32, device=self.device)
            states = torch.tensor(np.array(obs_buf), dtype=torch.float32, device=self.device)
            prob_list = self.actor(states)

            m = Categorical(prob_list)

            log_prob_tensor = m.log_prob(torch.tensor(act_buf, dtype=torch.int64, device=self.device))
            loss = torch.sum(-log_prob_tensor * R_t)  # TODO: add a baseline

            # gradient
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f'Epoch {epoch} Loss: {loss.item()}, Reward: {sum(rew_buf)}')
            self.trajectory_buffer.reset()
        return

def main():
    config = SimpleNamespace(
        env='CartPole-v1',
        batch=8,
        seed=0,
        num_epochs=1,
        sparsity=0.5,
        steps_per_epoch=5,
        device = 'cpu'
    )
    vpg = VPG(config)
    vpg.train_VPG()
    return

if __name__ == '__main__':
    main()