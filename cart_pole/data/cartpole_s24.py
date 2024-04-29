import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import tqdm

class CartPoleQLearning:
    
    """
    Q(S,A)←Q(S,A) +α[R+γmaxaQ(S′,a)−Q(S,A)]

    Algorithm parameters:  step size α∈(0,1], small ε >0
    Initialize Q(s,a), for all s∈S+, a∈A(s), arbitrarily except that Q(terminal,·) = 0
    Loop for each episode:
        Initialize S 
        Loop for each step of episode:
            Choose A from S using policy derived from Q(e.g.,ε-greedy)
            Take action A, observe R, S′
            Q(S,A)←Q(S,A) +α[R+γmaxaQ(S′,a)−Q(S,A)]
            S←S′; A←A′;
        until S is terminal
    """
    
    def __init__(self,
                 buckets,
                 decay,
                 min_epsilon,
                 min_alpha,
                 gamma,
                 environment,
                 min_velocity,
                 max_velocity):
        
        self.buckets = buckets
        self.decay = decay
        self.min_epsilon = min_epsilon
        self.min_alpha = min_alpha
        self.gamma = gamma
        self.environment = environment
        
        self.lower_bounds = [
            self.environment.observation_space.low[0],
            min_velocity,
            self.environment.observation_space.low[2],
            -math.radians(50)
        ]
        self.upper_bounds = [
            self.environment.observation_space.high[0],
            max_velocity,
            self.environment.observation_space.high[2],
            math.radians(50)
        ]
        
        self.action_value_table = None
        self.alpha = None
        self.epsilon = None
        self.episode_cnt = 0
        
        self.training_reward_history = []
        
        self.reset_action_value_table()
    
    
    def update_epsilon(self):
        """"""
        self.epsilon = max(self.min_epsilon, min(1., 1. - math.log10((self.episode_cnt + 1) / self.decay)))
    
    def update_alpha(self):
        """"""
        self.alpha = max(self.min_alpha, min(1., 1. - math.log10((self.episode_cnt + 1) / self.decay)))
    
    def reset_action_value_table(self):
        """Subroutine to reset the values in the action value table"""
        self.action_value_table = np.zeros(self.buckets + (self.environment.action_space.n, ))
    
    
    def state_to_tiles(self, s):
        """
        Args:
            s (array): an array representing the state. returned from self.environment.step(action)
            
        Returns:
            tuple of len 4 to help navigate through our action value table
        """
        tiles = list()
        
        for i in range(len(s)):
            scaling = (s[i] + abs(self.lower_bounds[i])) / (self.upper_bounds[i] - self.lower_bounds[i])
            scaled_s = int(round((self.buckets[i] - 1) * scaling))
            scaled_s = min(self.buckets[i] - 1, max(0, scaled_s))
            tiles.append(scaled_s)
        return tuple(tiles)
    
    def choose_action(self, state):
        """"""
        
        # Allow for some exploration...
        if np.random.random() <= self.epsilon:
            return self.environment.action_space.sample()
        else:
            return np.argmax(self.action_value_table[state])
        
    def q_learning_update(self, s, a, r, s_prime, a_prime):
        """"""
        self.action_value_table[s][a] += self.alpha * (r + self.gamma * max(self.action_value_table[s_prime]) 
                                                       - self.action_value_table[s][a])
        
    def train(self, n_episodes, reset):
        """"""
        if reset:
            self.reset_action_value_table()
            self.episode_cnt = 0
            
        for _ in range(n_episodes):
        
            # Resetting the environment and our episode counter
            s = self.state_to_tiles(self.environment.reset())

            # Resetting our alpha and epsilon
            self.update_epsilon()
            self.update_alpha()
            
            terminal_state = False
            
            episode_cumsum_r = 0
            
            while not terminal_state:
                a = self.choose_action(state=s)
                s_prime, r, terminal_state, _ = self.environment.step(action=a)
                episode_cumsum_r += r
                s_prime = self.state_to_tiles(s=s_prime)
                a_prime = self.choose_action(state=s_prime)
                self.q_learning_update(s=s, a=a, r=r, s_prime=s_prime, a_prime=a_prime)
                s = s_prime
                
            self.training_reward_history.append(episode_cumsum_r)
                
            # Updating our episode counter
            self.episode_cnt += 1
            
    def test(self, display):
        """""" 
        s = self.state_to_tiles(s=self.environment.reset())
        steps = 0
        terminal_state = False
        
        # We don't need to explore during testing - do we?
        self.epsilon = 0
        
        while not terminal_state:
            if display:
                self.environment.render()
                
            a = self.choose_action(state=s)
            s_prime, r, terminal_state, _ = self.environment.step(action=a)
            s_prime = self.state_to_tiles(s=s_prime)
            s = s_prime
            steps += 1
            
        if display:
            self.environment.close()
            
        return steps

def main():    
    np.random.seed(860720)

    cart_pole_qlearning = CartPoleQLearning(buckets=(1, 1, 6, 12),
                        decay=25,
                        min_epsilon=0.1,
                        min_alpha=0.1,
                        gamma=0.98,
                        environment=gym.make('CartPole-v0'),
                        min_velocity=-0.5,
                        max_velocity=0.5)

    cart_pole_qlearning.train(n_episodes=130, reset=True)

    cart_pole_qlearning.test(display=True)

    results = []

    for _ in range(100):
        results.append(cart_pole_qlearning.test(display=False))
        
    np.mean(results)