import numpy as np
import gym

class OffPolicyControl:

    def __init__(self, env: object, epsilon: float, Q: np.array, alpha: float, gamma: float,
                 reward_modified: dict = None):
        self.env = env
        self.epsilon = epsilon
        self.Q = Q
        self.alpha = alpha
        self.gamma = gamma
        self.reward_modified = reward_modified
        self.number_actions = Q.shape[1]
        self.reward_history = {}
        self.actions_dict = {0: 'rigth', 1: 'left', 2: 'up', 3: 'down', 4: 'go off'}

    def action_epsilon_greedy(self, state: int):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[state])
        return np.random.randint(self.number_actions)

    def update_value(self, state: int, action: int, reward: int, new_state: int, action_max: int):
        if self.reward_modified is None:
            reward = reward
        else:
            reward = self.reward_modified[reward]
        self.Q[state, action] = self.Q[state, action] + self.alpha * (
                reward + self.gamma * self.Q[new_state, action_max] - self.Q[state, action])

    def episode(self, episode_value: int):
        done = False
        state = self.env.reset()
        g = 0
        while not done:
            action = self.action_epsilon_greedy(state)
            new_state, reward, done, info = self.env.step(action)
            action_max = np.argmax(self.Q[new_state])
            self.update_value(state, action, reward, new_state, action_max)
            state = new_state
            g = + reward
            if episode_value % 10:
                self.reward_history[episode_value] = g

    def iter_episode(self, episodes: int):
        for i in range(episodes):
            self.episode(i)

    def select_polices(self):
        result = []
        list_board = self.env.board.tolist()
        flat_list = [item for sublist in list_board for item in sublist]
        for i in range(0, self.Q.shape[0]):
            if flat_list[i] == ' ':
                value = self.Q[i]
                value[value == 0] = -np.inf
                result.append(self.actions_dict[np.argmax(value)])
            else:
                result.append('*')
        return np.array(result).reshape(self.env.dimensions)

class MDPs:
    def __init__(self,
                 dimensions: tuple,
                 locked_cell: dict,
                 initial_rewards: dict,
                 chance_moving: dict,
                 be_same_place_bool: bool,
                 win_cell: tuple,
                 reward_move: dict,
                 initial_position: tuple = (0, 0),
                 ):

        """
         'rigth':0,
        'left':1,
        'up':2,
        'down':3,
        'go off':4
        """
        self.initial_position = initial_position
        self.chance_moving = chance_moving
        self.dimensions = dimensions
        self.initial_rewards = initial_rewards
        self.locked_cell = locked_cell
        self.be_same_place_bool = be_same_place_bool
        self.reward_move = reward_move
        self.win_cell = win_cell
        self.board = self.environment()

        self.list_index_board = self.map_index_tuple_board()
        self.current_state = self.list_index_board.index(initial_position)
    def map_index_tuple_board(self):
        n_rows, n_columns = self.dimensions
        list_index = []
        for row in range(n_rows):
            for columns in range(n_columns):
                list_index.append((row, columns))
        return list_index

    def reset(self):
        return self.list_index_board.index(self.initial_position)

    def environment(self) -> np.array:
        list_value = np.array([' '] * self.dimensions[0] * self.dimensions[1])
        self.board = list_value.reshape(self.dimensions)
        for location, value in self.locked_cell.items():
            self.board[location] = value
        return self.board

    def validate_locked_move(self, position: tuple):
        """
        Validar si las celdas están bloqueadas.
        """
        if self.board[position] == '*':
            return True
        else:
            return False

    def get_possible_actions(self, current_state):
        """
        Obtener las posiciones en las que nos podemos mover dado un punto inicial.
        """
        current_state = self.list_index_board[current_state]
        if current_state == self.win_cell:
            return {4: self.win_cell}
        else:
            row, column = current_state
            up = row - 1
            down = row + 1
            left = column - 1
            right = column + 1

            moving = {}
            if up < 0:
                moving[2] = current_state
            else:
                moving[2] = (up, column)
            if down > self.dimensions[0] - 1:
                moving[3] = current_state
            else:
                moving[3] = (down, column)
            if left < 0:
                moving[1] = current_state
            else:
                moving[1] = (row, left)
            if right > self.dimensions[1] - 1:
                moving[0] = current_state
            else:
                moving[0] = (row, right)
            movements={}
            for key, value in moving.items():
                if self.validate_locked_move(value):
                    value= current_state
                movements[key]=value
            #movements = {key: value for key, value in moving.items() if self.validate_locked_move(value)}
            if self.be_same_place_bool:
                return movements
            else:
                return {key: value for key, value in movements.items() if value != current_state}

    def step(self, action):
        info = ''
        if self.current_state == self.list_index_board.index(self.win_cell):
            done = True
            new_state = self.current_state
            reward = 100
            return new_state, reward, done, info
        else:
            new_state = self.get_possible_actions(self.current_state)[action]
            reward = -1
            done = False
            self.current_state = self.list_index_board.index(new_state)
            return self.list_index_board.index(new_state), reward, done, info


env_taxi = gym.make("Taxi-v3")
env_taxi.reset()
env_taxi.render()

nuumero_de_estados = env_taxi.observation_space.n
numero_acciones = env_taxi.action_space.n

q = np.ones((nuumero_de_estados, numero_acciones))

off_policy = OffPolicyControl(env=env_taxi,
                              epsilon=0.05,
                              Q=q,
                              alpha=0.001,
                              gamma=1,
                              reward_modified = {-1:1,20:5,-10:-10}
                             )

off_policy.iter_episode(episodes=5000)
print(off_policy.Q)