import numpy as np

class Env:
    def __init__(self, board, terminal_states: list, initial_state: tuple = None) -> None:
        self._board = board
        self.height = len(self._board)
        self.width = len(self._board[0])
        self.initial_state = initial_state
        self._terminal_states = terminal_states
        self.current_state = None; self._reset() # Gives a value to current state.
        self.actions = [0, 1, 2, 3]

    # Generates a random state that isn't a wall and isn't a terminal state.
    def _random_state(self):
        state = (np.random.randint(self.height), np.random.randint(self.width))
        while self._is_wall(state) or self._is_terminal(state):
            state = (np.random.randint(self.height), np.random.randint(self.width))
        return state

    # Reset the current state depending on if there was an initial state or not.
    def _reset(self):
        self.current_state = self.initial_state if self.initial_state != None else self._random_state()
    
    # Does an action over the environment. 
    # Returns (new_state, reward, terminal).
    # If new_state is terminal, resets the environment.
    def do_action(self, action: str) -> tuple:
        x, y = self.current_state
        terminal = False

        # Executes an operation over the current state according to the action selected.
        # UP
        if action == 0:
            x -= 1
        # DOWN
        elif action == 1:
            x += 1
        # LEFT
        elif action == 2:
            y -= 1
        # RIGHT
        elif action == 3:
            y += 1
        else:
            raise Exception("Selected non-existant action")
        
        # If the index goes out of bounds or goes to a wall, go back to the last state.
        if not (0 <= x < self.height and 0 <= y < self.width and not self._is_wall((x, y))):
            x, y = self.current_state

        # Assign the new state to be the current one.
        self.current_state = (x, y)

        # If the new state is terminal, reset the whole board.
        if self._is_terminal(self.current_state):
            terminal = True
            self._reset()

        # Return the reward according to the type of the cell.
        return (self.current_state, self._board[x][y], terminal)
        
    def _is_wall(self, state):
        x, y = state
        return self._board[x][y] == np.NINF

    def _is_terminal(self, state):
        return state in self._terminal_states
    
gridworld_board = np.zeros((10, 10))
gridworld_walls = [(2, 1), (2, 2), (2, 3), (2, 4), (2, 6), (2, 7), (2, 8),
                   (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)]
for wall in gridworld_walls:
    x, y = wall
    gridworld_board[x][y] = np.NINF
gridworld_board[4][5] = -1
gridworld_board[6][5] = 1
gridworld_board[7][5] = -1
gridworld_board[7][6] = -1

gridworld = Env(gridworld_board, [(6, 5)], (0, 0))


labyrinth_board = np.full((12, 11), -1.0)
for y in range(len(labyrinth_board[0])):
    labyrinth_board[0][y] = float("-inf")
    labyrinth_board[6][y] = float("-inf")
labyrinth_board[0][2] = -1.0
labyrinth_board[6][2] = -1.0
labyrinth_board[6][8] = -1.0
for x in range(len(labyrinth_board)):
    labyrinth_board[x][5] = float("-inf")
labyrinth_board[3][5] = -1.0
labyrinth_board[9][5] = -1.0
labyrinth_board[0][2] = 20

labyrinth = Env(labyrinth_board, [(0, 2)])

class TaxiEnv:
    # Houses has to have a length greater than one.
    def __init__(self, board, reward_dict: dict, houses: list) -> None:
        self._board = board
        self.height = len(self._board)
        self._reward_dict = reward_dict
        self.width = len(self._board[0])
        self.houses = houses
        self.actions = [0, 1, 2, 3, 4, 5] # UP, DOWN, LEFT, RIGHT, DROP, PICKUP.
        self.current_state = self._random_state()
        # Generate any house as passenger pickup place.
        self._passenger_pickup = self._new_random_destiny(self.current_state)
        # Generate any house as passenger dropoff place.
        self._passenger_dropoff = self._new_random_destiny(self._passenger_pickup)
        # Car does not have a passenger initially.
        self._has_passenger = 0

    # Generates a random state that isn't a wall and isn't a terminal state.
    def _random_state(self):
        state = (np.random.randint(self.height), np.random.randint(self.width))
        while self._is_wall(state) or self._is_house(state):
            state = (np.random.randint(self.height), np.random.randint(self.width))
        return state
    
    # Reset the current state depending on if there was an initial state or not.
    def _reset(self):
        self.current_state = self._random_state()
    
    def _new_random_destiny(self, state):
        new_destiny = np.random.choice(len(self.houses))
        while self.houses[new_destiny] == state:
            new_destiny = np.random.choice(len(self.houses))
        return self.houses[new_destiny]

    # Does an action over the environmt. 
    # Returns (new_state, reward, terminal).
    # If new_state is terminal, resets the environment.
    def do_action(self, action: str) -> tuple:
        x, y = self.current_state
        terminal = False
        # Reward by default.
        reward = self._reward_dict["step"]

        # Executes an operation over the current state according to the action selected.
        # UP
        if action == 0:
            x -= 1
        # DOWN
        elif action == 1:
            x += 1
        # LEFT
        elif action == 2:
            y -= 1
        # RIGHT
        elif action == 3:
            y += 1
        # DROP
        elif action == 4:
            if self.current_state == self._passenger_dropoff and self._has_passenger == 1:
                reward = self._reward_dict["dropoff"]
                self._has_passenger = 0
                terminal = True
            else: 
                reward = self._reward_dict["bad_action"]
        # PICKUP
        elif action == 5:
            if self.current_state == self._passenger_pickup and self._has_passenger == 0:
                reward = self._reward_dict["pickup"]
                self._has_passenger = 1
            else: 
                reward = self._reward_dict["bad_action"]
        else:
            raise Exception("Selected non-existant action")
        
        # If the index goes out of bounds or goes to a wall, go back to the last state.
        if not (0 <= x < self.height and 0 <= y < self.width and not self._is_wall((x, y))):
            x, y = self.current_state

        # Assign the new state to be the current one.
        self.current_state = (x, y)

        if terminal:
            self._reset()

        # Returns the current state, the reward, the destiny and if has passenger or not.
        return (self.current_state, reward, terminal, self._has_passenger)
        
    def _is_wall(self, state):
        x, y = state
        return self._board[x][y] == np.NINF

    def _is_house(self, state):
        return state in self.houses
    
    def _is_bridge(self, state):
        x, y = state
        return self._board[x][y] == np.INF
    

taxi_board = np.zeros((5, 8))
for x in range(2):
    taxi_board[x+3][1] = float("-inf")
    taxi_board[x][3] = float("-inf")
    taxi_board[x+3][5] = float("-inf")
for x in range(3):
    taxi_board[x][1] = float("inf")
    taxi_board[x+2][3] = float("inf")
    taxi_board[x][5] = float("inf")
rewards = {
    "step": -1, 
    "pickup": 10,
    "dropoff": 50,
    "bad_action": -100 
}
houses = [(0, 0), (0, 7), (4, 0), (4, 7)]
taxi_env = TaxiEnv(taxi_board, rewards, houses)