import argparse
from collections import namedtuple
from itertools import count

import gym
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Cart Pole

parser = argparse.ArgumentParser(description='')
parser.add_argument('--gamma', type=float, default=0.98,
                    metavar='G', help='default: 0.99')
parser.add_argument('--seed', type=int, default=523, metavar='N',
                    help='default: 543')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='default: 10')
args = parser.parse_args()


def objective(trial):
    learning_rate = trial.suggest_loguniform("lr", 1e-5, 1)
    gamma = trial.suggest_uniform('gamma', 0.7000, 0.9999)
    seed = trial.suggest_int('seed', 1, 1000)
    print(seed)
    nn_size = trial.suggest_categorical('nn_size', [64, 128, 256])
    np_float = trial.suggest_categorical(
        'np_float', ["float16", "float32", "float64"])

    env = gym.make('CartPole-v1')
    env.seed(seed)
    torch.manual_seed(seed)

    SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

    class Policy(nn.Module):
        """
        implements both actor and critic in one model
        """

        def __init__(self):
            super(Policy, self).__init__()
            self.affine1 = nn.Linear(4, nn_size)

            # actor's layer
            self.action_head = nn.Linear(nn_size, 2)

            # critic's layer
            self.value_head = nn.Linear(nn_size, 1)

            # action & reward buffer
            self.saved_actions = []
            self.rewards = []

        def forward(self, x):
            """
            forward of both actor and critic
            """
            x = F.relu(self.affine1(x))

            # actor: choses action to take from state s_t
            # by returning probability of each action
            action_prob = F.softmax(self.action_head(x), dim=-1)

            # critic: evaluates being in the state s_t
            state_values = self.value_head(x)

            # return values for both actor and critic as a tuple of 2 values:
            # 1. a list with the probability of each action over the action space
            # 2. the value from state s_t
            return action_prob, state_values

    model = Policy()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if np_float == "float16":
        eps = np.finfo(np.float16).eps.item()
    elif np_float == "float32":
        eps = np.finfo(np.float32).eps.item()
    elif np_float == "float64":
        eps = np.finfo(np.float64).eps.item()

    def select_action(state):
        state = torch.from_numpy(state).float()
        probs, state_value = model(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        model.saved_actions.append(
            SavedAction(m.log_prob(action), state_value))

        # the action to take (left or right)
        return action.item()

    def finish_episode():
        """
        Training code. Calculates actor and critic loss and performs backprop.
        """
        R = 0
        saved_actions = model.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in model.rewards[::-1]:
            # calculate the discounted value
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

        # reset gradients
        optimizer.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = torch.stack(policy_losses).sum() + \
            torch.stack(value_losses).sum()

        # perform backprop
        loss.backward()
        optimizer.step()

        # reset rewards and action buffer
        del model.rewards[:]
        del model.saved_actions[:]

    def main():
        running_reward = 10
        list_reward = []
        last_100 = 0

        # run inifinitely many episodes
        for i_episode in count(1):

            # reset environment and episode reward
            state = env.reset()
            ep_reward = 0

            # for each episode, only run 9999 steps so that we don't
            # infinite loop while learning
            for t in range(1, 10000):

                # select action from policy
                action = select_action(state)

                # take the action
                state, reward, done, _ = env.step(action)

                if args.render:
                    env.render()

                model.rewards.append(reward)
                ep_reward += reward
                if done:
                    break

            # update cumulative reward

            list_reward.append(ep_reward)

            if len(list_reward) >= 100:
                # print(sum(list_reward))
                last_100 = sum(list_reward[-100:])/100

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

            # perform backprop
            finish_episode()

            # log results
            # if i_episode % args.log_interval == 0:
            # print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
            #     i_episode, ep_reward, running_reward))

            if i_episode > 300:
                return i_episode
                break
            if last_100 >= 195:
                solved = i_episode - 100
                print('Solved at Episode {} With avg score of {} over the following 100 Episodes'.format(
                    solved, last_100))
                # print(last_100)
                # print(list_reward[-100:])
                # plt.plot(list_reward)
                # plt.show()
                return solved
                break
            # check if we have "solved" the cart pole problem
            if running_reward > env.spec.reward_threshold:
                # print("Solved! Running reward is now {} and "
                #       "the last episode runs to {} time steps!".format(running_reward, t))
                break
    return main()


storage = optuna.storages.RedisStorage(
    url='redis://34.123.159.224:6379/DB1',
)


study = optuna.create_study(
    study_name="cartpolwe", storage=storage, load_if_exists=True)
study.optimize(objective, n_trials=20)
print(study.best_params)