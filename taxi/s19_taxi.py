# !pip install gym
# !pip install pygame
import gym
import numpy as np
import random
# import pygame

env = gym.make("Taxi-v3")
state_space = env.observation_space.n
action_space = env.action_space.n
Q = np.zeros((state_space, action_space))

total_episodes = 2500        
total_test_episodes = 10      
max_steps = 200               
learning_rate = 0.01         
gamma = 0.99                  
epsilon = 1.0                
max_epsilon = 1.0             
min_epsilon = 0.001           
decay_rate = 0.01   

def epsilon_greedy_policy(Q, state):
  if(random.uniform(0,1) > epsilon):
    action = np.argmax(Q[state])
  else:
    action = env.action_space.sample()
  return action

penalties_per_episode = []
sucess_dropoff_per_episode = []

for episode in range(total_episodes):
    state = env.reset()
    step = 0
    penalties = 0
    dropoffs = 0
    done = False
    epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
    for step in range(max_steps):
        action = epsilon_greedy_policy(Q, state)
        new_state, reward, done, info = env.step(action)
        if reward == -10:
            penalties += 1
        if reward == 20:
            reward = 5
            dropoffs += 1
            
        Q[state][action] = Q[state][action] + learning_rate * (reward + gamma * 
                                    np.max(Q[new_state]) - Q[state][action])      
        if done == True: 
            break
        state = new_state
    penalties_per_episode.append(penalties)
    sucess_dropoff_per_episode.append(dropoffs)