import numpy as np
import time
import random

class GridEnvironment:
    def __init__(self, grid_height, grid_width):
        self.name = "Base Grid Environment"
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.grid = [[0] * grid_width for _ in range(grid_height)]
        self.start = (0, 0)
        self.goal = [(grid_height - 1, grid_width - 1)]
        
    def get_name(self):
        return self.name

    def get_grid(self):
        return self.grid
    
    def get_row_col(self, state):
        row, col = state
        return row, col
    
    def get_start(self):
        return self.start
    
    def reset(self):
        return self.get_start()
    
    def get_goal(self):
        return self.goal
    
    def get_grid_height(self):
        return self.grid_height
    
    def get_grid_width(self):
        return self.grid_width
        
    def get_actions(self, state):
        row, col = self.get_row_col(state)
        actions = []
        if row > 0 and self.grid[row - 1][col] is not None:
            actions.append('up')
        if row < self.get_grid_height() - 1 and self.grid[row + 1][col] is not None:
            actions.append('down')
        if col > 0 and self.grid[row][col - 1] is not None:
            actions.append('left')
        if col < self.get_grid_width() - 1 and self.grid[row][col + 1] is not None:
            actions.append('right')
        return actions
        
    def is_terminal(self, state):
        return state in self.get_goal()
        
    def step(self, state, action):
        row, col = self.get_row_col(state)
        if action == 'up':
            next_row = max(row - 1, 0)
            next_col = col
        elif action == 'down':
            next_row = min(row + 1, self.get_grid_height() - 1)
            next_col = col
        elif action == 'left':
            next_row = row
            next_col = max(col - 1, 0)
        elif action == 'right':
            next_row = row
            next_col = min(col + 1, self.get_grid_width() - 1)
        else:
            raise ValueError(f"Invalid action {action}")
        next_state = (next_row, next_col)
        reward = self.get_grid()[next_row][next_col]
        return next_state, reward
    
    def render(self):
        # Print the current state of the grid environment
        for row in range(self.grid_height):
            for col in range(self.grid_width):
                if (row, col) == self.start:
                    print('S', end=' ')
                elif (row, col) in self.goal:
                    print('G', end=' ')
                elif self.grid[row][col] is None:
                    print('X', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()

class GridWorld(GridEnvironment):
    def __init__(self, grid_height, grid_width, rewards, start, goal):
        super().__init__(grid_height, grid_width)
        self.name = "Grid World"
        self.grid = rewards
        self.start = start
        self.goal = goal


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
        for row in range(self.env.get_grid_height()):
            for col in range(self.env.get_grid_width()):
                state = (row, col)
                if state in self.env.get_goal():
                    self.q_table[state] = {"": self.env.get_grid()[row][col]}
                elif self.env.get_grid()[row][col] is None:
                    self.q_table[state] = None
                else:
                    self.q_table[state] = {}
                    for action in self.env.get_actions(state):
                            self.q_table[state][action] = 0

    def choose_action(self, state):

        if self.env.name == "Taxi":
            if self.env.is_pickup_stop(state):
                return "pickup"

        possible_state_actions = self.env.get_actions(state)
        if random.uniform(0, 1) < self.exploration_rate:
            # Choose a random action
            return random.choice(possible_state_actions)
        else:
            # Choose the best action according to the Q-table
            q_values = [self.q_table[state][action] for action in possible_state_actions]
            max_q = max(q_values)
            count = q_values.count(max_q)
            if count > 1:
                best_actions = [possible_state_actions[i] for i in range(len(possible_state_actions)) if q_values[i] == max_q]
                return random.choice(best_actions)
            else:
                return possible_state_actions[q_values.index(max_q)]

    def update_q_table(self, state, action, reward, next_state):
        old_q_value = float(self.q_table[state][action])
        next_max_q_value = 0
        for q_table_action in self.q_table[next_state]:
            if self.q_table[next_state][q_table_action] > next_max_q_value or next_max_q_value == 0:
                next_max_q_value = self.q_table[next_state][q_table_action]
            
        new_q_value = (1 - self.learning_rate) * old_q_value + self.learning_rate * (reward + self.discount_factor * next_max_q_value)
        # print(f"Old Q Value: {old_q_value} - New Q Value: {new_q_value} - Action: {action}")
        self.q_table[state][action] = new_q_value

    def estabilized(self, old_q, new_q):
        for state in old_q:
            if old_q[state] is None or new_q[state] is None:
                continue
            else:
                for action in old_q[state]:
                    if abs(old_q[state][action] - new_q[state][action]) > 0.0001:
                        return False
        return True
    
    def print_policy(self):
        for row in range(self.env.get_grid_height()):
            row_str = ""
            for col in range(self.env.get_grid_width()):
                state = (row, col)
                if state in self.env.get_goal():
                    row_str += "  G  "
                elif self.env.get_grid()[row][col] is None:
                    row_str += ("  X  ")
                else:
                    q_values = [self.q_table[state][action] for action in self.env.get_actions(state)]
                    max_q = max(q_values)
                    count = q_values.count(max_q)
                    policy_str = ""
                    if max_q != 0:
                        if count > 1:
                            best_actions = [self.env.get_actions(state)[i] for i in range(len(self.env.get_actions(state))) if q_values[i] == max_q]
                            policy_str += f"{random.choice(best_actions):>5}"
                        else:
                            policy_str += f"{self.env.get_actions(state)[q_values.index(max_q)]:>5}"
                        
                    row_str += f"{policy_str:>5}"
                row_str += " - "
            print(row_str)

    def run_episodes(self):
        end = False
        num_episodes = 0
        start_time = time.time()
        while not end and num_episodes < 1600:

            if num_episodes % 1 == 0 and self.env.get_name() == "Taxi":
                print(f"Running episode: {num_episodes}")
                # env.render()
            num_episodes += 1

            old_q = {}
            for state in self.q_table:
                old_q[state] = {}
                if self.q_table[state] is None:
                    old_q[state] = None
                else:
                    for action in self.q_table[state]:
                        old_q[state][action] = self.q_table[state][action]
            
            state = self.env.get_start()
            done = False

            start_time_episode = time.time()
            while not done:
                action = self.choose_action(state)
                next_state, reward = self.env.step(state, action)
                # print(f"State: {state} - Action: {action} - Reward: {reward} - Next State: {next_state}")
                self.update_q_table(state, action, reward, next_state)
                state = next_state

                if self.env.is_terminal(next_state):
                    done = True

                duration_episode = time.time() - start_time_episode
                # If duration is higher than 7 seconds, stop the episode
                if duration_episode > 7:
                    done = True

            
            end = self.estabilized(old_q, self.q_table)
            if (time.time() - start_time) > 180:
                end = True
            
            # print("--------------------")
            # print(f"Episode: {num_episodes}")
            # print(self.q_table)
            # print(self.env.get_goal())
            # print("--------------------\n")

            self.env.reset()


        print("\n--------------------")
        print(f"Number of episodes: {num_episodes}")
        # print(self.q_table)
        print(self.print_policy())
        print("--------------------\n")
    
    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward = self.env.step(state, action)
                if self.env.is_terminal(next_state):
                    done = True
                self.update_q_table(state, action, reward, next_state)
                state = next_state

    def test(self):
        state = self.env.reset()
        done = False
        while not done:
            action = np.argmax(self.q_table[state])
            next_state, reward = self.env.step(state, action)
            if self.env.is_terminal(next_state):
                done = True
            state = next_state
            self.env.render()


rewards = [[0, 0, 0, 1], [0, None, 0, -1], [0, 0, 0, 0]]
env = GridWorld(3, 4, rewards, (2, 0), [(0, 3), (1, 3)])
agent = QLearningAgent(env)
agent.run_episodes()

