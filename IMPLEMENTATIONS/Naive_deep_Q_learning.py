
import gym
import lbforaging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve 
import time


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
    
class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma = 0.99, 
                            epsilon = 1.0, eps_dec = 1e-5, eps_min = 0.1):
        
        self.lr  = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor(observation, dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state)
            action = T.argmax(actions).item() 
        else:
            action = np.random.choice(self.action_space)

        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec\
        if self.epsilon > self.eps_min else self.eps_min

    def learn(self, state, action, reward, state_):
        self.Q.optimizer.zero_grad()
        states = T.tensor(state, dtype= T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        states_ = T.tensor(state_, dtype=T.float).to(self.Q.device)

        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(states_).max()

        q_target = rewards + self.gamma*q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()

if __name__ =='__main__':
    env = gym.make("Foraging-8x8-2p-2f-v2") 
    n_games = 10000
    scores1 = []
    scores2 = []
    eps_history = []

    agent1 = Agent(12, 6, lr = 0.0001)
    agent2 = Agent(12, 6, lr = 0.0001)

    for i in range(n_games):
        score = [0, 0]
        done = [False, False]
        obs= env.reset()

        while  (done[0] and done[1])  == False:
            action = [0, 0]
            action[0] = agent1.choose_action(obs[0])
            action[1] = agent2.choose_action(obs[1])
            obs_, reward, done, info = env.step(tuple(action))
            score[0] = reward[0]
            score[1] =reward[1]
            agent1.learn(obs[0], action[0], reward[0], obs_[0])
            agent2.learn(obs[1], action[1], reward[1], obs_[1])
            obs = obs_

            if i % 100 == 0:
                env.render()
                time.sleep(1/60)

        scores1.append(score[0])
        scores2.append(score[1])
        eps_history.append(agent1.epsilon)

        if i  % 100 == 0:
            avg_score1 = np.mean(scores1[-100:])
            print('episode ', i, 'score1 %.1f avg score1 %.1f epsilon1%.2f' %(score[0], avg_score1, agent1.epsilon))
            avg_score2 = np.mean(scores2[-100:])
            print('episode ', i, 'score2 %.1f avg score2 %.1f epsilon2%.2f' %(score[1], avg_score2, agent2.epsilon))
            print("\n")
            print("\n")

    T.save(agent1.Q.state_dict(), 'a1_qn.pt')
    T.save(agent2.Q.state_dict(), 'a2_qn.pt')

    filename1 = 'Naive_dqn_a1.png'
    filename2 ='Naive_dqn_a2.png'

    x = [i+1 for i in range(n_games)]
    plot_learning_curve(x, scores1, eps_history, filename1)
    plot_learning_curve(x, scores2, eps_history, filename2)

