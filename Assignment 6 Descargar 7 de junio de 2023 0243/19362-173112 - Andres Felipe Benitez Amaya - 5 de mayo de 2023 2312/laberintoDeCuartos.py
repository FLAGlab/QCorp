import numpy as np

class RoomsEnvironment:
    def __init__(self):
        self.board = np.array([['W', 'W', 'W', 'W'],
                               ['W', 'R', '1', 'W'],
                               ['W', '2', '3', 'W'],
                               ['W', 'W', 'W', 'W']])
        self.dimensions = (4, 4)
        self.start_state = None

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        i, j = state
        actions = []
        if self.board[i-1, j] != 'W':  
            actions.append('up')
        if self.board[i+1, j] != 'W': 
            actions.append('down')
        if self.board[i, j-1] != 'W':  
            actions.append('left')
        if self.board[i, j+1] != 'W':  
            actions.append('right')
        return actions

    def do_action(self, action):
        i, j = self.current_state
        if action == 'up':
            new_state = (i-1, j)
        elif action == 'down':
            new_state = (i+1, j)
        elif action == 'left':
            new_state = (i, j-1)
        elif action == 'right':
            new_state = (i, j+1)

        self.current_state = new_state
        reward = self.get_reward(new_state)
        return reward, new_state

    def reset(self):
        self.start_state = self.get_random_state()
        self.current_state = self.start_state

    def is_terminal(self, state):
        i, j = state
        return self.board[i, j] == '1'

    def get_reward(self, state):
        i, j = state
        if self.board[i, j] == '1':
            return 1
        elif self.board[i, j] != 'W':
            return -0.1
        else:
            return -1

    def get_random_state(self):
        i, j = np.random.randint(1, 3, size=2)
        return (i, j)
    
rooms = np.array([[0, 1, 2, 3],
                  [4, 5, 6, 7],
                  [8, 9, 10, 11],
                  [12, 13, 14, 15]])

start_state = (3, 0)
goal_state = (0, 0)

def reward_function(state):
    if state == goal_state:
        return 0
    else:
        return -1
    
Q = np.zeros((16, 4))

alpha = 0.5
gamma = 1.0
epsilon = 0.1
num_episodes = 1000

for i in range(num_episodes):
    state = start_state
    done = False
    
    while not done:
        if np.random.random() < epsilon:
            action = np.random.choice([0, 1, 2, 3])
        else:
            action = np.argmax(Q[state])
        
        next_state = get_next_state(state, action)
        reward = reward_function(next_state)
        Q[state][action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])
        state = next_state
        
        if state == goal_state:
            done = True
    
    if i % 100 == 0:
        print(f"Episode {i}: {np.sum(Q)}")