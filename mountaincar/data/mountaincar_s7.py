from tile3 import IHT, tiles
import numpy as np
import argparse
import sys

import gym
from gym import wrappers, logger
from SarsaLambdaAgent import SarsaLambdaAgent
import numpy as np

TILES = 2048
NUM_TILINGS = 8
TILE_WIDTH = 0.125
ALPHA = 0.1/NUM_TILINGS
LAMDA = 0.9

class SarsaLambdaAgent():
    """
    Author: Anas Mohamed
    Email: amohamed@ualberta.ca

    Summary:
    A Sarsa(lambda) Agent with eligiblity tracing and tile coding.


    Sarsa Control Agent algorithm
    Reference: Section 12.7 from RL book (2nd edition) by Andrew Barto and Richard S. Sutton.

    Implements Tile coding using tile3 library.
    Author: Richard S. Sutton
    Reference: http://incompleteideas.net/tiles/tiles3.html

    """
    def __init__(self):
        self.prevState = None
        self.action = None
        self.features = dict()
        self.Xvector = None
        self.prevXvector = None
        self.Ztrace = np.zeros(TILES)
        self.iht = IHT(TILES)
        self.prevState = None
        self.weightVector = np.random.uniform(-0.001, 0,TILES)
        self.Q = dict()

    def start(self,observation):
        """
        Arguments:observation - numpy array
        Returns: action - integer
        Hint: Initialize the variables that you want to reset before starting
        a new episode, pick the first action, don't forget about exploring
        starts
        """
        self.Ztrace = np.zeros(TILES)
        # Randomly picks between 4-left and 6-right
        self.action = self.getGreedyAction(observation)

        self.prevState = observation

        return self.action

    def act(self, observation, reward, done):
        """
        Arguments: reward - floting point,observation - numpy array
        Returns: action - integer
        Hint: select an action based on pi
        """
        delta = reward
            
        # Generate features for prevState
        self.generateFeatures(self.prevState, self.action)
        prevFeatures = self.features[self.prevState[0], self.prevState[1], self.action]
        prevEstimate = np.sum(self.weightVector[prevFeatures])
        
        delta = np.subtract(delta, prevEstimate)
        self.Ztrace[prevFeatures] = 1

        # Update action values for prevState, action
        self.Q[self.prevState[0], self.prevState[1]][self.action] = prevEstimate


        # Get the next action
        self.action = self.getGreedyAction(observation)
        
        # Generate features for currentState
        self.generateFeatures(observation, self.action)
        currentFeatures = self.features[observation[0],observation[1], self.action]
        currentEstimate = np.sum(self.weightVector[currentFeatures])

        delta = np.add(delta, currentEstimate)
        
        # Update the weight vector
        deltaZ = np.multiply(delta, self.Ztrace)
        self.weightVector =  np.add(self.weightVector, np.multiply(ALPHA, deltaZ))

        # Update the replacing trace
        self.Ztrace = np.multiply(LAMDA, self.Ztrace)

        self.prevState =observation
        
        return self.action

    def end(self, reward):
        """
        Arguments: reward - floating point
        Returns: Nothing
        Hint: do necessary steps for policy evaluation and improvement
        """
        delta = reward
            
        # Generate features for prevState
        self.generateFeatures(self.prevState, self.action)
        prevFeatures = self.features[self.prevState[0], self.prevState[1], self.action]
        prevEstimate = np.sum(self.weightVector[prevFeatures])
        
        delta = np.subtract(delta, prevEstimate)
        self.Ztrace[prevFeatures] = 1

        # Update action values for prevState, action
        self.Q[self.prevState[0], self.prevState[1]][self.action] = prevEstimate

        # Update the weight vector
        deltaZ = np.multiply(delta, self.Ztrace)
        self.weightVector =  np.add(self.weightVector, np.multiply(ALPHA, deltaZ))

    def getGreedyAction(self, observation):
        if (observation[0],observation[1]) in self.Q:
            return np.argmax(self.Q[observation[0],observation[1]])
        else:
            self.Q[observation[0],observation[1]] = [0]*3
            self.generateFeatures(observation, 0)
            self.Q[observation[0],observation[1]][0] = self.weightVector[self.features[observation[0],observation[1], 0]].sum()
            self.generateFeatures(observation, 1)
            self.Q[observation[0],observation[1]][1] = self.weightVector[self.features[observation[0],observation[1], 1]].sum()
            self.generateFeatures(observation, 2)
            self.Q[observation[0],observation[1]][2] = self.weightVector[self.features[observation[0],observation[1], 2]].sum()
            return np.argmax(self.Q[observation[0],observation[1]])

    def generateFeatures(self,observation, action):
        # print(state[1])
        # print(action)
        if (observation[0],observation[1], action) not in self.features:
            positionScale = 8/(0.5+1.2)
            velocityScale = 8/(0.07+0.07)
            self.features[observation[0],observation[1], action] = tiles(self.iht, NUM_TILINGS, [observation[0]*positionScale,observation[1]*velocityScale], [action])
        else:
            pass

def video_schedule(episode_id):
    return episode_id % 50 == 0 or episode_id % 340 == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='MountainCar-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)

    env_orig = gym.make(args.env_id)

    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/SarsaControAgent-results'
    env = wrappers.Monitor(env_orig, directory=outdir, force=True, video_callable=video_schedule)
    env.seed(0)
    np.random.seed(0)
    agent = SarsaLambdaAgent()

    episode_count = 2000
    episode_rewards = np.zeros(episode_count)
    reward = -1
    done = False
    i = 0
    np.ALLOW_THREADS = 2
    while (i < episode_count):
        if i >= 100:
            avg = np.average(episode_rewards[i-100:i])
            if avg > -110.0:
                break
        totalReward = 0
        observation = env.reset()
        while True:
            if totalReward == 0:
                # For step 1, only
                action = agent.start(observation)
                observation, reward, done, _ = env.step(action)
            else:
                action = agent.act(observation, reward, done)
                observation, reward, done, _ = env.step(action)

            totalReward += reward

            if done:
                agent.end(reward)
                episode_rewards[i] = totalReward
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        i += 1

    np.save("rewardPerEpisode", episode_rewards[:i])
    # Close the env and write monitor result info to disk
    env.close()
