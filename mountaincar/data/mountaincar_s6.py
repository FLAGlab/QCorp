import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from IPython import display

from paiutils import neural_network as nn
from paiutils import reinforcement as rl
from paiutils import reinforcement_agents as ra


# see if using GPU and if so enable memory growth
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

genv = gym.make('MountainCar-v0')
max_steps = genv._max_episode_steps
print(max_steps)
print(genv.observation_space, genv.action_space)

class GymWrapper(rl.GymWrapper):
    def render(self):
        """Render for Jupyter Notebooks.
        """
        x = self.genv.render(mode='rgb_array')
        if 'img' in self.__dict__:
            self.img.set_data(x)
            plt.axis('off')
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            self.img = plt.imshow(x)

env = GymWrapper(genv)

inputs = keras.layers.Input(shape=env.state_shape)
x = nn.dense(64)(inputs)
x = nn.dense(64)(x)
outputs = nn.dense(env.action_size, activation='softmax',
                   batch_norm=False)(x)
amodel = keras.Model(inputs=inputs,
                     outputs=outputs)
amodel.compile(optimizer=keras.optimizers.Adam(.001),
                loss='mse')
amodel.summary()

inputs = keras.layers.Input(shape=env.state_shape)
x = nn.dense(64)(inputs)
x = nn.dense(64)(x)
outputs = keras.layers.Dense(1)(x)

cmodel = keras.Model(inputs=inputs,
                     outputs=outputs)
cmodel.compile(optimizer=keras.optimizers.Adam(.001),
               loss='mse')
cmodel.summary()

discounted_rate = .99
lambda_rate = 0.95
agent = ra.A2CAgent(
    amodel, cmodel, discounted_rate,
    lambda_rate=lambda_rate,
    create_memory=lambda shape, dtype: rl.Memory(20000)
)

def end_episode_callback(episode, step, reward):
    global agent
    if reward > -150:
        # Check if agent may be able
        # to solve the environment
        old_playing_data = agent.playing_data
        agent.set_playing_data(training=False,
                               memorizing=False)
        result = env.play_episodes(
            agent, 10, max_steps,
            verbose=False, episode_verbose=False,
            render=False
        )
        print(f'Validate Results: {result}')
        if result >= -110:
            agent.save(
                save_dir, note=f'A2C_{episode}_{result}'
            )
        agent.playing_data = old_playing_data
        if result >= -100:
            # end early
            return True

# No warmup needed
#agent.set_playing_data(memorizing=True, verbose=True)
#env.play_episodes(agent, 100, max_steps, random=True,
#                  verbose=True, episode_verbose=False,
#                  render=False)

agent.set_playing_data(
    training=True, memorizing=True,
    batch_size=16, mini_batch=1024,
    epochs=1, repeat=5,
    entropy_coef=0,
    verbose=False
)
save_dir = ''
num_episodes = 500
result = env.play_episodes(
    agent, num_episodes, max_steps,
    verbose=True, episode_verbose=False,
    render=False,
    end_episode_callback=end_episode_callback
)
agent.save(save_dir, note=f'A2C_{ndx}_{result}')