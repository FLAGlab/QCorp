import numpy as np
import gym
import matplotlib.pyplot as plt
# %matplotlib inline
from IPython import display
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def build_model(input_size, output_size,learning_rate,final_activation,nodes):
    model = Sequential()
    model.add(Dense(nodes, input_dim=input_size, activation="relu"))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dense(nodes, activation="relu"))
    model.add(Dense(output_size, activation='linear'))
    if final_activation is not None:
        model.add(final_activation)
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    return model

def plot_results(x,y,x_label,y_label,title):
    """
    Plot results for each function
    :param x: [] list of values to display along X axis
    :param y: [] list of values to display  along Y axis
    :param x_label: Str name of the x axis
    :param y_label: Str name of the y axis
    :param title : Str title of the chart
    """
    plt.plot(x,y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()
    plt.close()

def QlearningNN(env, epsilon_decay, discount, epsilon, min_eps, episodes, model, other_stats):
    local_stats = {'reached_goal': 0, 'current_moves': 0, 'moves_to_success': [], 'highest_position': -1.2,
                   'largest_velocity': -0.07}

    # Initialize variables to track rewards
    reward_list = []
    ave_reward_list = []

    # Run Q learning algorithm
    for i in range(episodes):
        # Initialize parameters
        done = False
        tot_reward, reward = 0, 0
        state = env.reset()

        steps = 0
        while done != True:
            # Render environment for last five episodes
            # if i >= (episodes - 20):
            #    env.render()
            steps += 1
            # Decrease epsilon up to epsilon_decay value 
            epsilon *= epsilon_decay
            epsilon = max(min_eps, epsilon)
            Q = model.predict(state.reshape(1, len(state)))
            # Determine next action based on epsilon 
            if np.random.rand() <= epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q)

            # Get next state and reward
            state2, reward, done, info = env.step(action)

            if (state2[0] > local_stats['highest_position']):
                local_stats['highest_position'] = state2[0]

            if (state2[1] > local_stats['largest_velocity']):
                local_stats['largest_velocity'] = state2[1]

            # Count only moves
            if action != 1:
                local_stats['current_moves'] += 1

            # Check if we reached the goal 
            if done and state2[0] >= 0.5:
                local_stats['reached_goal'] += 1
                local_stats['moves_to_success'].append(local_stats['current_moves'])
                local_stats['current_moves'] = 0
                target = reward


            # Adjust Q value for current state
            else:
                pred = model.predict(state2.reshape(1, len(state2)))
                Q_next = np.max(pred)
                target = reward + Q_next * discount

            Q[0][action] = target
            tot_reward += reward

            if steps > 199:
                done = True

            # Train model with state and action 
            model.fit(x=state.reshape(1, len(state)), y=Q, epochs=1, verbose=0)

        # Track rewards
        reward_list.append(tot_reward)

        if (i + 1) % 100 == 0:
            ave_reward = np.mean(reward_list)
            ave_reward_list.append(ave_reward)
            reward_list = []
            other_stats['reached_goal_episodes'].append(local_stats['reached_goal'])
            other_stats['highest_positions'].append(local_stats['highest_position'])
            other_stats['largest_velocities'].append(local_stats['largest_velocity'])
            if (len(local_stats['moves_to_success']) > 0):
                other_stats['ave_moves_to_success'].append(np.mean(local_stats['moves_to_success']))
            else:
                other_stats['ave_moves_to_success'].append(0)
            local_stats = {'reached_goal': 0, 'current_moves': 0, 'moves_to_success': [], 'highest_position': -1.2,
                           'largest_velocity': -0.07}

        if (i + 1) % 100 == 0:
            print('Episode {} Average Reward: {}'.format(i + 1, ave_reward))

    env.close()
    return ave_reward_list, other_stats

