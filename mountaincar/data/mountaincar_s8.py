import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
import os

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.env= env
        self.state_size= state_size
        self.action_size= action_size
        
        self.memory= deque(maxlen=200000)
        
        self.gamma= 0.99
        
        self.epsilon = 1.0
        self.epsilon_decay= .85
        self.epsilon_min=0.00001
        
        self.learning_rate= 0.001251
        self.model= self._build_model()
        self.target_model=self._build_model()
        
        self.update_target_model()
        
    def _build_model(self):
        model= tf.keras.models.Sequential()
        state_shape= self.env.observation_space.shape
        model.add(tf.keras.layers.Dense(24, input_shape= state_shape, activation='relu'))
        model.add(tf.keras.layers.Dense(48, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        
        return model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand(1) <=self.epsilon:
            return random.randrange(self.action_size)
        act_values= self.model.predict(state)
        return np.argmax(act_values[0])
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states=[]
        targets=[]
        
        for state, action, reward, next_state, done in minibatch:
            target=reward
            if not done:
                target = (reward + self.gamma * np.amax(self.target_model.predict(next_state)[0]))
            target_f= self.model.predict(state)
            target_f[0][action]= target
            states.append(state[0])
            targets.append(target_f[0])
            
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)
            
        
    def load(self, name):
        self.model.load_weights(name)
    def save(self, name):
        self.model.save_weights(name)

def main():
    done = False
    counter=0 
    scores_memory= deque(maxlen=100)
    for e in range(n_episodes):
        state=env.reset()

        state= np.reshape(state, [1, state_size])
        
        for time in range(7000):
            if e % 50==0:
                env.render()
            action= agent.act(state)
            next_state, reward, done, halp =env.step(action)
            
            next_state = np.reshape(next_state, [1, state_size])

            
            agent.remember(state, action, reward, next_state, done)
                
            if len(agent.memory)>batch_size:
                agent.replay(batch_size)

            
            state = next_state

            if done:
                scores_memory.append(time)
                scores_avg= np.mean(scores_memory)*-1

                
                print('episode: {}/{}, score: {}, e {:.2}, help: {}, reward: {}, 100score avg: {}'.format(e, n_episodes, time, agent.epsilon, state, reward, scores_avg))

                break
        agent.update_target_model()
            
            
        if agent.epsilon > agent.epsilon_min:
                agent.epsilon *= agent.epsilon_decay
            
        if e % 50==0:
            agent.save(output_dir + 'weights_final' + '{:04d}'.format(e) + ".hdf5")
