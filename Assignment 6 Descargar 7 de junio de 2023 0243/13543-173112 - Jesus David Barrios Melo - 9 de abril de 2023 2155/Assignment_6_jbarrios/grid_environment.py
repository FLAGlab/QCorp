import numpy as np

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

class RoomMaze(GridEnvironment):
    def __init__(self, reward=5):
        # For this environment, the first row is full of spaces that cannot be accessed. This is included only to be able to include the final state, which is outside the environment grid.
        super().__init__(11, 10)
        self.name = "Room Maze"
        # Set start as a random position in the grid
        self.start = (np.random.randint(1, 11), np.random.randint(0, 10))
        for col in range(10):
            self.grid[0][col] = None
        self.goal = [(0, 2)]
        self.grid[0][2] = reward

    def reset(self):
        self.start = (np.random.randint(1, 11), np.random.randint(0, 10))
        return self.get_start()

    def get_actions(self, state):
        row, col = self.get_row_col(state)
        actions = []
        if row > 0 and self.grid[row - 1][col] is not None:
            if row != 6 and row != 1:
                actions.append('up')
            elif row == 6 and col in [2, 7]:
                actions.append('up')
            elif row == 1 and col == 2:
                actions.append('up')
        if row < self.get_grid_height() - 1 and self.grid[row + 1][col] is not None:
            if row != 5 or col in [2, 7]:
                actions.append('down')
        if col > 0 and self.grid[row][col - 1] is not None:
            if col != 5 or row in [3, 7]:
                actions.append('left')
        if col < self.get_grid_width() - 1 and self.grid[row][col + 1] is not None:
            if col != 4 or row in [3, 7]:
                actions.append('right')
        return actions

class Taxi(GridEnvironment):
    def __init__(self, stops):
        super().__init__(5, 5)
        self.name = "Taxi"
        self.start = (np.random.randint(0, 5), np.random.randint(0, 5))
        self.stops = stops
        self.passenger_position = self.stops[np.random.randint(0, len(self.stops))]
        self.passenger_picked_up = False
        self.goal = []

    def is_pickup_stop(self, state):
        return (state == self.passenger_position and not self.passenger_picked_up)
    
    def reset(self):
        self.start = (np.random.randint(0, 5), np.random.randint(0, 5))
        self.passenger_position = self.stops[np.random.randint(0, len(self.stops))]
        self.passenger_picked_up = False
        self.goal = []
        return self.get_start()

    def get_actions(self, state):
        row, col = self.get_row_col(state)
        actions = []
        if row > 0 and self.grid[row - 1][col] is not None:
            actions.append('up')
        if row < self.get_grid_height() - 1 and self.grid[row + 1][col] is not None:
            actions.append('down')
        if col > 0 and self.grid[row][col - 1] is not None:
            if col not in [1,2,3]:
                actions.append('left')
            elif col == 1 and row not in [3,4]:
                actions.append('left')
            elif col == 2 and row not in [0,1]:
                actions.append('left')
            elif col == 3 and row not in [3,4]:
                actions.append('left')
        if col < self.get_grid_width() - 1 and self.grid[row][col + 1] is not None:
            if col not in [0,1,2]:
                actions.append('right')
            elif col == 0 and row not in [3,4]:
                actions.append('right')
            elif col == 1 and row not in [0,1]:
                actions.append('right')
            elif col == 2 and row not in [3,4]:
                actions.append('right')
        actions.append('pickup')
        actions.append('dropoff')
        return actions

    def step(self, state, action):
        row, col = self.get_row_col(state)
        
        if action == 'up':
            next_row = max(row - 1, 0)
            next_col = col
            reward = self.get_grid()[next_row][next_col]
        elif action == 'down':
            next_row = min(row + 1, self.get_grid_height() - 1)
            next_col = col
            reward = self.get_grid()[next_row][next_col]
        elif action == 'left':
            next_row = row
            next_col = max(col - 1, 0)
            reward = self.get_grid()[next_row][next_col]
        elif action == 'right':
            next_row = row
            next_col = min(col + 1, self.get_grid_width() - 1)
            reward = self.get_grid()[next_row][next_col]
        elif action == 'pickup':
            if self.passenger_position == (row, col) and not self.passenger_picked_up:
                self.passenger_picked_up = True
                reward = 1
                for stop in self.stops:
                    row_stop, col_stop = stop
                    passenger_row, passenger_col = self.passenger_position
                    if stop != (passenger_row, passenger_col):
                        self.goal.append(stop)
                for stop in self.goal:
                    row_stop, col_stop = stop
                    self.grid[row_stop][col_stop] = 5
            else:
                reward = -10
            next_row = row
            next_col = col
        elif action == 'dropoff':
            if state in self.goal and self.passenger_picked_up:
                    reward = 5
            else:
                reward = -10
            next_row = row
            next_col = col
        else:
            raise ValueError(f"Invalid action {action}")
        next_state = (next_row, next_col)
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
                elif (row, col) == self.passenger_position:
                    print('P', end=' ')
                elif (row, col) in self.stops:
                    print('D', end=' ')
                else:
                    print('.', end=' ')
            print()
        print()