#imports 
# need to add the rest of the CSV code



import gym
import lbforaging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import time
import collections
from ENV_AGENT import*
import csv

def get_dist(obs):
    distance = abs(obs[6] - obs[9]) + abs(obs[7] - obs[10])
    return distance
# Main loop
if __name__ =='__main__':
    
    print("main loop starts \n")
    #environment init
    env = gym.make("Foraging-8x8-2p-2f-v2") # init env
    #agent Init
    agent1 = Agent(12, 6, lr = 0.0001)
    agent2 = Agent(12, 6, lr = 0.0001)
    #Buffer parameters
    capacity = 20000 #buffer capacity
    min_buff_size = 10000
    batch_size = 40 # more of a training parameter 
    #buffer init
    buffer1 =  ExperienceReplay(capacity) # buffer list with 2 buffers 
    buffer2 = ExperienceReplay(capacity)
    
  
    #load previously trained models 
    agent1.Q_primary.load_state_dict(T.load('a1.1_DQN_withbuffer.pt'))
    agent1.Q_primary.train()
    agent2.Q_primary.load_state_dict(T.load('a2.1_DQN_withbuffer.pt'))
    agent2.Q_primary.train()
    
    
    #n_games = 10000 # no of episode

    # Variables to store results 
    #Episode individual scores 
    ep_scores1 = []
    ep_scores2 = []
    ep_scores1_stock = []
    ep_scores2_stock = []
    #summed ep scores
    ep_scores = []
    #epsilon history
    eps_history = [] # epsilon history same for both
    
    #no. of episodes
    frame_idx = 0

    #reset obs 
    obs = env.reset() # init environment
    done = [False, False] # True when one ep finishes 
# no. of trainning cycles 
    training_id = 0
#temp variables to keep track of rewards in one episode
#are reset to 0 before the start of every episode
    ep_score = 0
    scores1 = 0
    scores2 = 0
    scores1_stock = 0
    scores2_stock = 0
    
# variables to keep track  of mean rewards 
    mean_ep_scores = 0
    last_mean_score = 0
    ctr = 0 # very unsophisticated method for checking if rewards are stable for a while 

n_col = 0
#INIT CSV stuff 
with open('frame_rewards.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['Frame', 'Reward1', 'Reward2'])  # Writing headers for the columns

# Start of loop; It terminates when converged ==> determined by a janky IF block
    while True:
        if (done[0] and done[1])  == True:
            frame_idx += 1
            obs = env.reset()
            ep_scores.append(ep_score)
            ep_scores1.append(scores1)
            ep_scores2.append(scores2)
            ep_scores1_stock.append(scores1_stock)
            ep_scores2_stock.append(scores2_stock)
            csvwriter.writerow([frame_idx, scores1_stock, scores2_stock])
            mean_ep_scores = np.mean(ep_scores[-100:])
            #print("\n ep-score list ----------------===========>>>>>>>> ",ep_scores)
            print("\n episode number >>>>>>>>>>>>>..     ", frame_idx)
            #print("len ep scores >>>>>>>>>>>>>>>>>>>   ", len(ep_scores))
            print("mean episode score: --------------", mean_ep_scores)
            print("\n epsilons ==================", agent1.epsilon)
            print ("foods collected so far ------", n_col)
            ep_score = 0
            scores1 = 0
            scores2 = 0
            scores1_stock = 0
            scores2_stock = 0
           # print("\n new episode ---------------- ")
            
            # The janky code to determine convergence 
            if frame_idx > 100:
                if last_mean_score < mean_ep_scores:
                    last_mean_score = mean_ep_scores
                    if last_mean_score > 20:
                        T.save(agent1.Q_primary.state_dict(), 'a1.1_DQN_withbuffer.pt')
                        T.save(agent2.Q_primary.state_dict(), 'a2.1_DQN_withbuffer.pt')
                        print("training completed in %d learning cycles", training_id)
                        ctr +=1
                        if ctr > 10:
                            print("training completed in %d learning cycles", training_id)
                            break

        action = [0, 0]  # list of actions
        # sample actions
        action[0] = agent1.choose_action(obs[0])
        action[1] = agent2.choose_action(obs[1])

            # take action
        obs_, reward, done, info = env.step(tuple(action))
        scores1_stock += reward[0]
        scores2_stock += reward[1]

        if reward[0] >0.1 and reward[1] > 0:
            n_col +=1
        
        #reward shaping
        for i in range(2):
            reward[i] = reward[i]*100 
            if action[i] != 0:
                reward[i] = reward[i] - 2
            else:
                reward[i] = reward[i] - 1
            if get_dist(obs[0]) < 4:
                reward[i] += 0.5
        

  
    #Render every so many episodes
       
        if mean_ep_scores > 0 and frame_idx % 50 == 0:
            env.render()
            time.sleep(1/60)


        agent1.decrement_epsilon()
        agent2.decrement_epsilon()
        #update epsiodic scores
        scores1 += reward[0] #for agent1
        scores2 += reward[1] #for agent2 
        ep_score += reward[0] + reward[1] #total reward 

        #store in exp_rep buffer
        exp1 = Experience(obs[0], action[0], reward[0], done[0], obs_[0])
        exp2 = Experience(obs[1], action[1], reward[1], done[1], obs_[1])
        buffer1.append(exp1)
        buffer2.append(exp2)
        #update obs with new obs 
        obs = obs_
        #check if buffer has enough experiences 
        if buffer1.__len__()> min_buff_size:
            #take a minibatch
            batch1 = buffer1.sample(batch_size)
            batch2 = buffer2.sample(batch_size)
            agent1.learn(batch1, batch_size, training_id)
            agent2.learn(batch2, batch_size, training_id)

            training_id += 1 # update training id
    # Reduce Rewards maybe 
    # fix agent values!!! 
    #everything else seems to be working 