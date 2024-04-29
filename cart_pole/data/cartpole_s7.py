import warnings
warnings.filterwarnings('ignore')

# Dependencies
import os
import math
from collections import deque
from chainer import functions as F
import skopt
import bottleneck as bn
import matplotlib.pyplot as plt

def learn_to_balance(model_layers=3,
                     model_layer_size=34,
                     model_layer_taper=1.0490184900205404,
                     model_activation_1=2,
                     model_activation_2=2,
                     model_activation_3=1,
                     mm_omega=3.3475360424700393,
                     batch_size=113,
                     memory_size=10000,
                     discount_factor=0.9725879085561605,
                     epsilon=0.9985988622823487,
                     epsilon_decay=0.9469565368958073,
                     lr=0.000584644333009868,
                     epsilon_min=0.01,
                     train_start=256,
                     n_episodes=500,
                     n_win_ticks=195,
                     n_avg_scores=100,
                     n_max_steps=200,
                     logging_int=10,
                     verbose=False,
                     render=False,
                     return_history=False,
                     win_100_scalar=1.1, # Optimizer bonus for a fast win < 100 episodes
                     win_10_scalar=1.15, # Optimizer bonus for a fast win < 10 episodes
                     seed=123):

  # Environment
  import gym
  env = gym.make('CartPole-v0')
  state_size = env.observation_space.shape[0]
  action_size = env.action_space.n

  # Reproducibility
  os.environ['PYTHONHASHSEED'] = str(seed)
  import random
  import numpy as np
  env.seed(seed)
  env.action_space.seed(seed)
  random.seed(seed)
  np.random.seed(seed)

  # Apply seed to tensorflow session
  import tensorflow as tf
  import keras.backend as K
  from keras.utils.generic_utils import get_custom_objects
  from keras.layers import Dense, Activation
  from keras.models import Sequential
  from keras.optimizers import Adam
  tf.reset_default_graph()
  session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
  tf.set_random_seed(seed)
  sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
  K.set_session(sess)

  # Mish Activation implementation
  # Source: https://github.com/digantamisra98/Mish/blob/master/Mish/Keras/mish.py
  class Mish(Activation):
    def __init__(self, activation="Mish", **kwargs):
      super(Mish, self).__init__(activation, **kwargs)
      self.__name__ = 'Mish'

  def mish(x):
    return x*K.tanh(K.softplus(x))

  get_custom_objects().update({'Mish': Mish(mish)})

  # Mellowmax: Softmax alternative
  # See: http://arxiv.org/abs/1612.05628
  # Source: https://github.com/chainer/chainerrl/blob/master/chainerrl/functions/mellowmax.py
  def mellowmax(values, omega=1., axis=1):
    n = values.shape[axis]
    return (F.logsumexp(omega * values, axis=axis) - np.log(n)) / omega

  # Build model
  def build_model(state_size, action_size, model_layers=3, model_activation_1=2, model_activation_2=2, model_activation_3=2, model_layer_size=96, model_layer_taper=0.5, lr=0.003):
    model = Sequential()

    model.add(Dense(int(model_layer_size), input_dim=state_size, kernel_initializer='he_uniform'))

    for i in range(int(model_layers)):
      layer_size = min(16, int(model_layer_size * (model_layer_taper ** i)))
      model.add(Dense(layer_size, kernel_initializer='he_uniform'))
      if i % 3 == 0:
        model_activation = model_activation_3
      elif i % 2 == 0:
        model_activation = model_activation_2
      else:
        model_activation = model_activation_1
    
      if round(model_activation) == 0:
        model.add(Activation('relu'))
      elif round(model_activation) == 1:
        model.add(Activation('tanh'))
      else:
        model.add(Mish())
    model.add(Dense(action_size, kernel_initializer='he_uniform'))
    model.compile(Adam(lr=lr), loss='mse')
    return model

  # Training
  # Source: https://github.com/yanpanlau/CartPole/blob/master/DQN/CartPole_DQN.py
  def get_action(state, action_size, model, epsilon):
    return np.random.randint(action_size) if np.random.rand() <= epsilon else np.argmax(model.predict(state)[0])

  def train_replay(memory, batch_size, train_start, discount_factor, mm_omega, model):
    if len(memory) < train_start:
      return
    minibatch = random.sample(memory,  min(int(batch_size), len(memory)))

    # Experience replay
    state_t, action_t, reward_t, state_t1, terminal = zip(*minibatch)
    state_t = np.concatenate(state_t)
    state_t1 = np.concatenate(state_t1)
    targets = model.predict(state_t)
    Q_sa = model.predict(state_t1)
    mm = mellowmax(Q_sa, omega=mm_omega).data

    targets[range(int(batch_size)), action_t] = reward_t + discount_factor * mm * np.invert(terminal)
    model.train_on_batch(state_t, targets)

  # Model
  model = build_model(state_size, action_size,
                      model_layers=model_layers, model_layer_size=model_layer_size,
                      model_layer_taper=model_layer_taper, lr=lr)

  # Training
  solution = []
  all_scores = []
  scores = deque(maxlen=int(n_avg_scores))
  memory = deque(maxlen=int(memory_size))
  solution_window_start = n_episodes

  for e in range(n_episodes):
    done = False
    score = 0
    step = 0
    state = env.reset()
    state = np.reshape(state, [1, state_size])

    while not done and step < n_max_steps:
      action = get_action(state, action_size, model, epsilon)
      next_state, reward, done, info = env.step(action)
      next_state = np.reshape(next_state, [1, state_size])

      memory.append(
          (state, action, reward if not done else -100, next_state, done))
      if epsilon > epsilon_min:
        epsilon *= epsilon_decay  # Decrease randomness
      train_replay(memory, batch_size, train_start, discount_factor, mm_omega, model)
      score += reward
      step += 1
      state = next_state

      if render:
        env.render()

      if done:
        env.reset()
        scores.append(score)
        all_scores.append(score)
        avg_score = np.mean(scores)

        if len(solution) == 0 and avg_score >= n_win_ticks and e >= n_avg_scores:
          # The start of a 100-episode averaging window with a mean score >= 195
          solution_window_start = e - n_avg_scores

          # The first episode score >= 195
          solution_episode_idx = next(
              x[0] for x in enumerate(all_scores) if x[1] >= n_win_ticks)

          solution.append(solution_window_start)
          print('Solved! Avg. reward >= 195.0 over 100 consecutive trials reached at episode {} \o/'.format(
              solution_window_start))
          print('First score >= 195 reached at episode {}.'.format(
              solution_episode_idx))

        if verbose > 0 and e % logging_int == 0:
          avg_display = '{:.2f}'.format(avg_score)
          print('[Episode {}] Average Score: {} | Total Rewards: {:.2f}'.format(
              e, avg_display, score))
  return solution_window_start, np.mean(all_scores), all_scores

def run_game(**config):
  solution_window_start, avg_score, all_scores = learn_to_balance(**config)
  if 'return_history' in config and config['return_history']:
        return solution_window_start, avg_score, all_scores
  if solution_window_start < 10:
    return avg_score * config['win_10_scalar']
  elif solution_window_start < 100:
    return avg_score * config['win_100_scalar']
  return avg_score

# Bounded region of parameter space
SPACE = [skopt.space.Real(0.0005, 0.01, name='lr', prior='uniform'),
         skopt.space.Real(0.9, 1.0, name='discount_factor', prior='uniform'),
         skopt.space.Real(0.9, 0.99, name='epsilon_decay', prior='uniform'),
         skopt.space.Real(0.9, 1.0, name='epsilon', prior='uniform'),
         skopt.space.Real(1.0, 30.0, name='mm_omega', prior='uniform'),
         skopt.space.Real(0.5, 1.5, name='model_layer_taper', prior='uniform'),
         skopt.space.Integer(16, 128, name='model_layer_size'),
         skopt.space.Integer(0, 2, name='model_activation_1'),
         skopt.space.Integer(0, 2, name='model_activation_2'),
         skopt.space.Integer(0, 2, name='model_activation_3'),
         skopt.space.Integer(32, 128, name='batch_size'),
         skopt.space.Integer(3, 5, name='model_layers')]

@skopt.utils.use_named_args(SPACE)
def objective(**params):
    return -1.0 * run_game(**params, win_100_scalar=1.1, win_10_scalar=1.2)

print('Searching parameter space... Now would be a good time to make coffee. ☕')

results = skopt.gbrt_minimize(objective, SPACE, n_calls=500, callback=[skopt.callbacks.VerboseCallback(n_total=500)], random_state=123)
print(results)
