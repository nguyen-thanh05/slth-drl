import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt


class LinearDQN:
    def __init__(self, config):
        self.gamma = config.gamma if hasattr(config, 'gamma') else 0.95
        self.epsilon = config.epsilon if hasattr(config, 'epsilon') else 1
        self.epsilon_decay_min = config.epsilon_decay_min if hasattr(config, 'epsilon_decay_min') else 0.05
        self.epsilon_decay = config.epsilon_decay_max if hasattr(config, 'epsilon_decay_max') else 0.995
        self.replay_buffer = ReplayMemory(config.replay_buffer_size if hasattr(config, 'replay_buffer_size') else 2000)
        # TODO: Implement n-step learning
        #self.n_step = config.n_step if hasattr(config, 'n_step') else 1
        self.batch_size = config.batch_size if hasattr(config, 'batch_size') else 32
        self.learning_rate = config.learning_rate if hasattr(config, 'learning_rate') else 0.00025
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.env = config.env
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        type_of_nn = config.type_of_nn if hasattr(config, 'type_of_nn') else 'linear'
        if type_of_nn == 'linear':
            self.policy_net = Linear_Net(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
            self.target_net = Linear_Net(self.state_dim, self.action_dim, config.hidden_dim).to(self.device)
        elif type_of_nn == 'conv2d':
            pass
        elif type_of_nn == 'linear_slth':
            pass
        elif type_of_nn == 'conv2d_slth':
            pass
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate, eps=1.5e-4)

    def select_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            action = self.env.action_space.sample()
        else:
            action = torch.argmax(self.policy_net(state.unsqueeze(0)), dim=1).item()
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
        
    def optimize_on_batch(self):
        states, actions, rewards, next_states, dones = zip(*self.replay_buffer.sample(128))
        states_batch = torch.stack(states)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states_batch = torch.stack(next_states)
        dones = torch.tensor(dones, device=self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(states_batch).gather(1, actions.unsqueeze(1))
        # Compute Q(s', a')
        with torch.no_grad():
            policy_next_a = self.policy_net(next_states_batch).argmax(1)
            next_q_values = self.target_net(next_states_batch).gather(1, policy_next_a.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        loss = F.smooth_l1_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_dqn(self, num_episodes):
        self.cummulative_reward_plot = []
        for episode in range(num_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device)
            cummulative_reward = 0
            done = False
            loss = None
            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                cummulative_reward += reward
                done = terminated or truncated
                done = int(done)
                next_state = torch.tensor(next_state, device=self.device)
                
                self.store_transition(state, action, reward, next_state, done)
                state = next_state
                
                if len(self.replay_buffer) > 150:
                    loss = self.optimize_on_batch()

                # Update target network
                target_state_dict = self.target_net.state_dict()
                policy_state_dict = self.policy_net.state_dict()
                for key in target_state_dict:
                    target_state_dict[key] = policy_state_dict[key] * 0.001 + target_state_dict[key] * 0.999
                self.target_net.load_state_dict(target_state_dict)
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_decay_min)
            self.cummulative_reward_plot.append(cummulative_reward)
            print(f'Episode: {episode}, Loss: {loss}, CummulativeReward: {cummulative_reward}')
            
        self.plot_cummulative_reward()
                    
    def plot_cummulative_reward(self):
        plt.plot(self.cummulative_reward_plot)
        plt.xlabel('Episodes')
        plt.ylabel('Cummulative Reward')
        plt.show()
                

class Linear_Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(Linear_Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x + F.relu(self.fc2(x))
        x = self.out(x)
        return x

class ReplayMemory:
    def __init__(self, max_size):
        self.buffer = [None] * max_size
        self.max_size = max_size
        self.index = 0
        self.size = 0
        self.len = 0

    def append(self, obj):
        self.buffer[self.index] = obj
        self.size = min(self.size + 1, self.max_size)
        self.index = (self.index + 1) % self.max_size
        self.len = self.len + 1 if self.len < self.max_size else self.max_size

    def sample(self, batch_size):
        indices = random.sample(range(self.size), batch_size)
        return [self.buffer[index] for index in indices]

    def __len__(self):
        return self.len
