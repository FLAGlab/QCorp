import itertools
import numpy as np
import gym
np.random.seed(0)
env = gym.make('CartPole-v0')
env.seed(0)

class Agent:
    def decide(self, observation):
        position, velocity, angle, angle_velocity = observation
        action = int(3. * angle + angle_velocity > 0.)
        return action

def play_once(env, agent, render=False, verbose=False):
    observation = env.reset()
    episode_reward = 0.
    for step in itertools.count():
        if render:
            env.render()
        action = agent.decide(observation)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            break
    if verbose:
        print('get {} rewards in {} steps'.format(
                episode_reward, step + 1))
    return episode_reward

agent = Agent()

episode_rewards = [play_once(env, agent) for _ in range(100)]
print('average episode rewards = {}'.format(np.mean(episode_rewards)))


episode_rewards = [play_once(env, agent) for _ in range(100000)]
print('average episode rewards = {}'.format(np.mean(episode_rewards)))

env.close()
