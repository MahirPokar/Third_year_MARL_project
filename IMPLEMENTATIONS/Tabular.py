import gym
import lbforaging
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
import csv

style.use("ggplot")

SIZE = 20
worldSize = 100
HM_EPISODES = 2
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 1
EPS_DECAY = 0.999998  # Every episode will be epsilon*EPS_DECAY
SHOW_EVERY = 1 # how often to play through env visually.

start_q_table = None# None or Filename

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1  # player key in dict
FOOD_N = 2  # food key in dict
ENEMY_N = 3  # enemy key in dict

def save_q_table(Q_table, filename='q_table.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(Q_table, f)

# Load Q_table from a file
def load_q_table(filename='q_table.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)

"""
if start_q_table is None:
    # initialize the q-table#
    q_table = {}
    for i in range(2):
        for ii in range(-SIZE+1, SIZE):
            for iii in range(-SIZE+1, SIZE):
                q_table[(i, ii)] = [np.random.uniform(-5, 0) for i in range(5)]
    q_table[None] = [np.random.uniform(-5, 0) for i in range(1,5)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

"""
# can look up from Q-table with: print(q_table[((-9, -2), (3, 9))]) for example

episode_rewards = []
env = gym.make("Foraging-8x8-2p-2f-v2")
Q_table = load_q_table('q_table.pkl')
action = [0,0]
epsilon = 0
alpha = 0.1
gamma = 0.99
ep_no = 0
save_ctr = 10000
score = [[], []]
ep_score = [0, 0]

with open('frame_rewards_tabular.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Frame', 'Reward1', 'Reward2']) 

while(1):
    obs = env.reset()
    obs = (tuple(obs[0]), tuple(obs[1]))
    done = [False, False]
    with open('frame_rewards_tabular.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([ep_no, ep_score[0], ep_score[1]]) 

        
    for i in range(2):
        score[i].append(ep_score[i])
        ep_score[i] = 0
    mean_score = np.mean(ep_score[-100:]) + np.mean(ep_score[-100:])
    print("mean_score",mean_score)
    ep_no += 1
    print(ep_no)
    if epsilon > 0.1:
        epsilon -= 0.9/50000
    if ep_no %100000 == 0:
        save_q_table(Q_table, 'q_table.pkl')
        save_ctr =  save_ctr + save_ctr/2
    while (done[0] and done[1] )== False:
        for i in range(2):
            if obs[i] not in Q_table[i]:
                Q_table[i][obs[i]] = [np.random.uniform(-5, 0) for i in range(6)]
            else:
                if np.random.rand() < epsilon:
                    action[i] = np.random.randint(6)
                else:
                    action[i] = np.argmax(Q_table[i][obs[i]])
        
        obs_, reward, done, info = env.step(tuple(action))
        
        obs_ = (tuple(obs_[0]), tuple(obs_[1]))
        for i in range(2):
            if obs_[i] not in Q_table[i]:
                Q_table[i][obs_[i]] = [np.random.uniform(-5, 0) for i in range(6)]
                ep_score[i] += reward[i]
                reward[i] = reward[i]*100
                reward[i] -= 1

        for i in range(2):
            old_q_value = Q_table[i][obs[i]][action[i]]
            if done[i] != True:
                next_max_q_value = np.max(Q_table[i][obs_[i]])
            else: 
                next_max_q_value = 0
            new_q_value = (1 - alpha) * old_q_value + alpha * (reward[i] + gamma * next_max_q_value)
            Q_table[i][obs[i]][action[i]] = new_q_value

        obs = obs_

"""
    if ep_no % 1000 == 0:  # Close and reopen the file to ensure data is saved
        csvfile.close()
        with open('frame_rewards_tabular.csv', 'a', newline='') as csvfile_append:
            csvwriter = csv.writer(csvfile_append)
            for row in score[0]:
                csvwriter.writerow([ep_no - len(score[0]) + score[0].index(row), row, score[1][score[0].index(row)]])
        csvfile = open('frame_rewards_tabular.csv', 'a', newline='')  # Reopen the file for future writes

"""        
