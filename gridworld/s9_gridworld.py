import numpy as np
import matplotlib.pyplot as plt

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


class QLearning:
    def __init__(self, step_size: float, discount: float, epsilon: float, env: Env) -> None:
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.env = env
        # This is not efficient because it stores non useful states.
        # Initialize Q as a Numpy array of 0s for each state.
        self.Q = np.zeros((self.env.height, self.env.width, len(self.env.actions)))
        self.last_state = self.env.current_state

    # Off course I assume there is at least one action to be made.
    def argmaxA_Q(self, state):
        x, y = state
        max_Q = float("-inf")
        maxs = []
        for action in self.env.actions:
            if self.Q[x][y][action] > max_Q:
                maxs = [action]
                max_Q = self.Q[x][y][action]
            elif self.Q[x][y][action] == max_Q:
                maxs.append(action)
        return np.random.choice(maxs)

    # Executes one step of the episode. Returns whether or not S' (the new state) is terminal.
    # Uses episode_num to make alpha converge.
    def episode_step(self, episode_num, converges):
        # "Old" state.
        x, y = self.last_state

        # Choose A from S using policy derived from Q (e.g., E-greedy).
        probability = np.random.random()
        if probability < self.epsilon:
            action = np.random.choice(self.env.actions)
        else:
            action = self.argmaxA_Q(self.last_state)
        
        # Take action A, observe R, S'.
        (new_state, reward, terminal) = self.env.do_action(action)
        nx, ny = new_state

        old_Q = self.Q[x][y][action]
        # Update Q.
        # * (1/episode_num) would make the series converge with certainty.
        self.Q[x][y][action] = self.Q[x][y][action] \
            + self.step_size \
                *(reward + self.discount*self.Q[nx][ny][self.argmaxA_Q((nx, ny))] - self.Q[x][y][action])
        
        # Determines if the Qvalues converge.
        new_Q = self.Q[x][y][action]
        if not (abs(new_Q) * 0.99 < abs(old_Q) < abs(new_Q) * 1.01):
            converges = converges & False

        # Update S <- S'.        
        self.last_state = new_state

        return terminal, converges
        
    def control(self):
        episode_num = 0
        converges = False
        while not converges:
            terminal = False
            converges = True
            while(not terminal):
                terminal, converges = self.episode_step(episode_num+1, converges)
            self.last_state = self.env.current_state
            episode_num += 1
        return episode_num
    
    def generate_policy(self):
        policy = np.full((self.env.height, self.env.width), len(self.env.actions))
        for x in range(self.env.height):
            for y in range(self.env.width):
                if (self.Q[x][y] == [0 for _ in self.env.actions]).all():
                    continue
                best_actions = []
                best_Q = float("-inf")
                for action in self.env.actions:
                    if self.Q[x][y][action] > best_Q:
                        best_actions = [action]
                        best_Q = self.Q[x][y][action]
                    elif self.Q[x][y][action] == best_Q:
                        best_actions.append(action)
                policy[x][y] = np.random.choice(best_actions)
        for state in self.env._terminal_states:
            x, y = state
            policy[x][y] = len(self.env.actions) + 1 # Just used to notice which is the exit.
        return policy
    
    def plot_scenario(self, title):
        plt.rcParams.update({'font.size': 7})
        policy = self.generate_policy()
        _, axes = plt.subplots()
        axes.set_title(title)
        axes.set_aspect('equal')
        axes.set_xlim(0, self.env.width)
        axes.set_ylim(self.env.height, 0)
        for y in range(self.env.width):
            for x in range(self.env.height):
                cell_policy = policy[x][y]
                text = ""
                if cell_policy == len(self.env.actions)+1:
                    color = 'lawngreen'
                    cell_qvalue = 0
                    text = "q: 0"
                elif cell_policy == len(self.env.actions):
                    color = 'dimgray'
                    cell_qvalue = float("-inf")
                else:
                    color = 'white'
                    cell_qvalue = round(self.Q[x][y][cell_policy], 2)
                    text = f"{policy_to_string(cell_policy)}\n{cell_qvalue}"

                if title == "Gridworld" and (x, y) in [(4, 5), (7, 5), (7, 6)] and color != 'dimgray':
                    color = 'pink'

                axes.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=color))
                axes.text(y + 0.5, x + 0.5, text, ha='center', va='center')
        plt.show()

def policy_to_string(policy):
    if policy == 0:
        return "∧"
    elif policy == 1:
        return "v"
    elif policy == 2:
        return "<"
    elif policy == 3:
        return ">"
    else:
        return ""

def main():
    agent2 = QLearning(0.1, 0.9, 0.9, gridworld)
    print(f"Episode number Gridworld: {agent2.control()}")
    agent2.plot_scenario("Gridworld")
