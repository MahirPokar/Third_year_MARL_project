import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time

SIZE = 8
worldSize = 8
HM_EPISODES = 500000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 1
EPS_DECAY = 0.9998  
SHOW_EVERY = 2000

start_q_table = 'qtable-1712863280.pickle'

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER1_N = 1  # player 1 key in dict
PLAYER2_N = 2  # player 2 key in dict
FOOD_N = 3  # food key in dict
LOAD_ACTION = 4  # Load action key in dict

# the dict!
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255),
     4: (255, 0, 0)}
class Blob:
    def __init__(self, player_num):
        self.worldSize = SIZE
        self.player_num = player_num
        self.x = np.random.randint(0, self.worldSize)
        self.y = np.random.randint(0, self.worldSize)

    def action(self, choice):
        if choice == 0:
            self.move(x=0, y=0)
        elif choice == 1:
            self.move(x=1, y=0)
        elif choice == 2:
            self.move(x=-1, y=0)
        elif choice == 3:
            self.move(x=0, y=1)
        elif choice == 4:
            self.move(x=0, y=-1)

    def move(self, x, y):
        self.x += x
        self.y += y
        self.x = np.clip(self.x, 0, self.worldSize - 1)
        self.y = np.clip(self.y, 0, self.worldSize - 1)

if start_q_table is None:
    q_table = {}
    for i in range(SIZE):
        for j in range(SIZE):
            for k in range(SIZE):
                for l in range(SIZE):
                    for m in  range(SIZE):
                        for n in range(SIZE):
                            q_table[((i, j), (k, l), (m,n))] = [np.random.uniform(-5, 0) for _ in range(25)]

else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []

for episode in range(HM_EPISODES):
    player1 = Blob(player_num=1)
    player2 = Blob(player_num=2)
    food = Blob(player_num=3)

    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(100):
        obs = ((player1.x, player1.y), (player2.x, player2.y), (food.x, food.y))

        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 25)

        player1.action(action % 5)  # Player 1's action
        player2.action(action // 5)  # Player 2's action

        
        if (player1.x == food.x and player1.y == food.y) and (player2.x == food.x and player2.y == food.y):
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
       

        new_obs = ((player1.x, player1.y), (player2.x, player2.y), (food.x, food.y))
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        
        q_table[obs][action] = new_q
   

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to blue color
            env[player1.x][player1.y] = d[PLAYER1_N]  # sets the player 1 tile to orange
            env[player2.x][player2.y] = d[PLAYER2_N]  # sets the player 2 tile to green
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD:  # Crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        episode_reward += reward
        if reward == FOOD_REWARD:
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()

with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
