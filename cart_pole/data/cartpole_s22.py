import gym
from gym import wrappers
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
from IPython.display import HTML
import os
# %matplotlib inline

def train_dqn(env, params, monitor=False, save_vid_every_n_episodes=50, max_episodes=2000, state_dict=None):
    """
    Args:
        params (dict): If this is set, load the agent with these params.
        env (gym.core.Env): The environment to train on
        monitor (Optional[bool, False]): Whether or not to monitor the environment and save a video
        save_vid_every_n_episodes (Optional[int, 50]): Only matters if monitor=True. This is the frequency at which to save videos.
        max_episodes (Optional[int, 2000]): Max number of episodes to try to solve an environment before giving up.
        state_dict (Optional[dict, None]): If this is set, the agent loads a state_dict, i.e. the parameters of the neural network. Mainly used if you want to load an agent later and save a video of best performance.
    """
    agent = DQNAgent(states, actions, params)
    scores = []
    scores_window = deque(maxlen=100)
    eps = params['eps_start']
    if monitor:
        env = wrappers.Monitor(env, 
                               directory='./monitors', 
                               video_callable=lambda episode_id: episode_id % save_vid_every_n_episodes == 0,
                               force=True)
    if state_dict:
        agent.policy_network.load_state_dict(state_dict)
    for episode in range(max_episodes):
        state = env.reset()
        score = 0.0
        for t in range(1000):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(params['eps_end'], eps*params['eps_decay'])
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode+1, np.mean(scores_window)), end="")
        if episode % 100 == 0 and episode > 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(scores_window)))
        if np.mean(scores_window)>=195.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(max(episode-100+1, 1), np.mean(scores_window)))
            break
    
    return scores, agent