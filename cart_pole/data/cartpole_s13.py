import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras

from paiutils import neural_network as nn
from paiutils import reinforcement as rl


def main():

    # see if using GPU and if so enable memory growth
    gpus = tf.config.list_physical_devices('GPU')
    print(gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    genv = gym.make('CartPole-v0')
    max_steps = genv._max_episode_steps
    print(max_steps)
    print(genv.observation_space, genv.action_space)

    env = rl.GymWrapper(genv)

    x0 = keras.layers.Input(shape=env.state_shape)
    x = nn.dense(32)(x0)
    x1 = nn.dense(16)(x)
    x2 = nn.dense(16)(x)
    #outputs = keras.layers.Dense(action_shape[0])(x)
    outputs = rl.DQNAgent.get_dueling_output_layer(
        env.action_size, dueling_type='avg'
    )(x1, x2)
    qmodel = keras.Model(inputs=x0,
                        outputs=outputs)
    qmodel.compile(optimizer=keras.optimizers.Adam(.001),
                loss='mse')
    qmodel.summary()

    policy = rl.StochasticPolicy(
        rl.GreedyPolicy(),
        rl.ExponentialDecay(.5, .1, .01, step_every_call=False),
        0, env.action_size
    )
    discounted_rate = .99
    agent = rl.DQNAgent(
        policy, qmodel, discounted_rate,
        enable_target=False, enable_double=False,
        enable_per=False
    )

    # Warmup
    agent.set_playing_data(memorizing=True, verbose=True)
    env.play_episodes(agent, 8, max_steps, random=True,
                    verbose=True, episode_verbose=False,
                    render=False)

    agent.set_playing_data(
        training=True, memorizing=True,
        learns_in_episode=False, batch_size=16,
        mini_batch=0, epochs=1, repeat=50,
        target_update_interval=1, tau=1.0,
        verbose=False
    )
    save_dir = ''
    num_episodes = 6
    for ndx in range(1):
        print(f'Save Loop: {ndx}')
        result = env.play_episodes(
            agent, num_episodes, max_steps,
            verbose=True, episode_verbose=False,
            render=False
        )
        agent.save(save_dir, note=f'DQN_{ndx}_{result}')

    agent.set_playing_data(training=False,
                        memorizing=False)
    step, total_reward = env.play_episode(
        agent, max_steps,
        verbose=True, render=False
    )
    print(total_reward)