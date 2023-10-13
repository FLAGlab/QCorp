import numpy as np
import random

class Taxi:
    def __init__(self):
        self.grid_size = 5
        self.num_states = self.grid_size**2 * 4 * 3
        self.num_actions = 6
        self.R = np.zeros((self.num_states, self.num_actions))
        self.Q = np.zeros((self.num_states, self.num_actions))
        self.passenger_location = None
        self.destination = None
        self.current_state = None

    def reset(self):
        self.passenger_location = random.randint(0, 3)
        self.destination = random.randint(0, 3)
        while self.destination == self.passenger_location:
            self.destination = random.randint(0, 3)
        x = random.randint(0, self.grid_size - 1)
        y = random.randint(0, self.grid_size - 1)
        direction = random.randint(0, 3)
        self.current_state = self.get_state(x, y, direction, self.passenger_location)
    
    def get_state(self, x, y, direction, passenger_location):
        state = passenger_location * self.grid_size**2 * 4
        state += x * self.grid_size * 4
        state += y * 4
        state += direction
        return state
    
    def get_reward(self, next_state):
        if self.R[next_state].max() < 0:
            return -10
        
        elif self.passenger_location == self.destination:
            return 5
        
        else:
            if self.passenger_location == next_state // (self.grid_size**2 * 4):
                return 1
        
            elif self.destination == next_state // (self.grid_size**2 * 4):
                return 5
            
            else:
                return 0
    
    def take_action(self, action):
        x, y, direction, passenger_location = self.get_state_vars(self.current_state)
        
        
        if action == 0:
            x = max(0, x - 1)
        elif action == 1:
            x = min(self.grid_size - 1, x + 1)
        elif action == 2:
            y = max(0, y - 1)
        elif action == 3:
            y = min(self.grid_size - 1, y + 1)
        elif action == 4:
            direction = (direction + 1) % 4
        elif action == 5:
            direction = (direction - 1) % 4
        
        if passenger_location == self.current_state // (self.grid_size**2 * 4):
            passenger_location = self.current_state // (self.grid_size**2 * 4)
        
        next_state = self.get_state(x, y, direction, passenger_location)
        
        reward = self.get_reward(next_state)
        return next_state, reward
    
    def get_state_vars(self, state):
        direction = state % 4
        y = (state // 4) % self.grid_size
        x = ((state // 4) - y) // self.grid_size
        passenger_location = (state // (self.grid_size**2 * 4)) % 4
        return x, y, direction, passenger_location