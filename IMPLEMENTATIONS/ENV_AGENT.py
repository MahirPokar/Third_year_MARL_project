
import gym
import lbforaging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import time
import collections


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super(LinearDeepQNetwork, self).__init__()

        self.fc1 = nn.Linear(input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions 
    

Experience = collections.namedtuple('Experience',  field_names= ['state', 'action', 'reward', 'done', 'new_state'])
class ExperienceReplay:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self, experience):
        #print(experience)
        self.buffer.append(experience)
  
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, dones, next_states = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


class Agent():
    def __init__(self, input_dims, n_actions, lr,   gamma = 0.999, 
                            epsilon = 1.0, eps_dec = 0.25e-5, eps_min = 0.1):
        
        self.lr  = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q_target = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        self.Q_primary = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)
        

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q_primary.device)
            actions = self.Q_primary.forward(state)
            action = T.argmax(actions).item() 
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec\
        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, batch, batch_size, training_id):
        sync_rate = 1000
        self.Q_primary.optimizer.zero_grad()
        self.Q_target.optimizer.zero_grad()

        state, action, reward, done, state_ = batch

        state = T.tensor(state, dtype=T.float, requires_grad=True).to(self.Q_primary.device)  # Set requires_grad to True
        action = T.tensor(action, requires_grad=False).to(self.Q_primary.device)  # No need for gradients on actions
        reward = T.tensor(reward, requires_grad=False).to(self.Q_primary.device)  # No need for gradients on rewards
        state_ = T.tensor(state_, dtype=T.float, requires_grad=True).to(self.Q_primary.device)  # Set requires_grad to True

        y = []
        q_pred = []

        for i in range(batch_size):
            q_pred.append(self.Q_primary.forward(state[i])[action[i]].unsqueeze(0))  # Unsqueeze to make it a proper tensor
            if done[i] == True:
                y.append(reward[i])
            else:
                y.append(reward[i] + self.gamma * self.Q_target.forward(state_[i]).max())

        y = T.tensor(y, dtype=T.float, requires_grad=False).to(self.Q_primary.device)  # No need for gradients on y

        loss = self.Q_primary.loss(y, T.cat(q_pred)).to(self.Q_primary.device)  # Use cat to concatenate q_pred

        loss.backward()
        self.Q_primary.optimizer.step()

        if training_id % sync_rate == 0:
            self.Q_target.load_state_dict(self.Q_primary.state_dict())
            self.Q_target.eval()
