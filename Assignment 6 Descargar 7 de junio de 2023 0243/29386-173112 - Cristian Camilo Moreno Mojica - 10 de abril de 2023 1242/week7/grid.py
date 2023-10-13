import numpy as np


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
