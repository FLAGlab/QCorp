import tensorflow as tf
tf.__version__

from keras.engine.base_layer import Layer
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.backend as K


import gym
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import matplotlib.pyplot as plt

class Mish(Activation):
    '''
    Mish Activation Function.
    .. math::
        mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + e^{x}))
    Shape:
        - Input: Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
        - Output: Same shape as the input.
    Examples:
        >>> X = Activation('Mish', name="conv1_act")(X_input)
    '''

    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'


def mish(x):
    return x*K.tanh(K.softplus(x))

get_custom_objects().update({'Mish': Mish(mish)})

class Model:
  def __init__(self):
    self.model = self.create_model()

  def create_model(self):
    model = Sequential()

    model.add(Dense(128, input_dim=len(env.observation_space.high), kernel_initializer='he_normal'))
    model.add(Mish(mish))
    # model.add(Activation('softplus'))

    model.add(Dense(64, kernel_initializer='he_normal'))
    model.add(Mish(mish))
    # model.add(Activation('tanh'))

    model.add(Dense(64, kernel_initializer='he_normal'))
    model.add(Mish(mish))
    # model.add(Activation('tanh'))

    model.add(Dense(env.action_space.n, kernel_initializer='he_normal'))
    # model.add(Activation('softmax'))

    model.compile(loss='mse', optimizer=Adam())
    return model
  
  def get_action(self, state):
    return np.argmax(self.predict(state)) if np.random.random() > epsilon else np.random.randint(env.action_space.n)

  def train(self, x_train, y_train):
    return self.model.train_on_batch(x_train, y_train)
  
  def predict(self, x):
    return self.model.predict(x)
  
  def get_weights(self):
    return self.model.get_weights()

  def set_weights(self, other):
    return self.model.set_weights(other.model.get_weights())
  
  def summary(self):
    return self.model.summary()
     
def get_state(st):
  return np.reshape(st, (1, len(env.observation_space.high)))

def train_the_model():
  if len(memory) < 1000:
    return

  batch = random.sample(memory,min(len(memory), BATCH_SIZE))

  x=[]
  y=[]
  for state, action, reward, next_state, done in batch:
    max_future_q = np.max(target_model.predict(next_state))
    new_q = reward + DISCOUNT * max_future_q * np.invert(done)

    current_q = model.predict(get_state(state))
    current_q[0][action] = new_q

    x.append(state)
    # print(f"x:{len(x)}")
    y.append(current_q)
    # print(f"y:{len(y)}")

  x = np.reshape(np.array(x), (-1, len(env.observation_space.high)))
  y = np.reshape(np.array(y), (-1, 2))
  model.train(x, y)

env = gym.make('CartPole-v0')

seed = 2
env.seed(seed)
random.seed(seed)
np.random.seed(seed)

EPISODES = 1000
DIM = len(env.observation_space.high)
DISCOUNT = 0.99
SHOW_AT = 100
show = False
BATCH_SIZE = 32

epsilon = 0.99
EPSILON_DECAY = 0.99
MIN_EPSILON = 0.01

memory = deque(maxlen=10000)
batch = deque(maxlen=BATCH_SIZE)
scores = deque(maxlen=100)
episode_rewards = []
avg_scores = []

task_complete = False

model = Model()
target_model = Model()
model.summary()

for episode in range(EPISODES):

  if not (episode % SHOW_AT):
    show = True
  else:
    show = False

  episode_reward = 0
  state = get_state(env.reset())
  
  done = False
  while not done:

      action = model.get_action(state)
      next_st, reward, done, _ = env.step(action)
      next_state = get_state(next_st)
      episode_reward += reward

      memory.append((state, action, reward if not done else -100, next_state, done))
      train_the_model()

      if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
      
      state = next_state
      
      if show:
        env.render()

      if done:
        env.reset()
        target_model.set_weights(model)
        episode_rewards.append(episode_reward)
        scores.append(episode_reward)
        avg_score = np.mean(scores)
        avg_scores.append(avg_scores)

        if not task_complete and avg_score >= 195 and episode >= 100:
          solution_episode = next(x[0] for x in enumerate(episode_rewards) if x[1] >= 195) 
          print(f"Solved at episode no. {solution_episode}")
          task_complete = True
        
        if not (episode % 10):
          print(f"Episode: {episode}:- average reward = {avg_score}, max reward till now = {np.max(episode_rewards)}")
     