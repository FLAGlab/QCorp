import random
import numpy as np

class Environment:
    def __init__(self, dims: tuple, current_state: tuple = (0, 0)):
        self.board = np.full(dims, ' ', dtype=object)
        self.dimensions = dims
        self.current_state = current_state
        self.terminals = set()
        self.actions = ["up", "down", "left", "right"]
        self.terminal_actions = ["out"]

    def get_current_state(self) -> tuple:
        return self.current_state

    def get_actions(self, pos: tuple) -> list:
        if pos in self.terminals:
            return self.terminal_actions
        return self.actions

    def get_possible_actions(self, pos: tuple) -> list:
        if pos in self.terminals:
            return self.terminal_actions
        possible_actions, (row, col) = [], pos
        if pos in self.terminals:
            return ["out"]
        if row > 0 and self.board[row - 1, col] != '*':
            possible_actions.append('up')
        if row < self.dimensions[0] - 1 and self.board[row + 1, col] != '*':
            possible_actions.append('down')
        if col > 0 and self.board[row, col - 1] != '*':
            possible_actions.append('left')
        if col < self.dimensions[1] - 1 and self.board[row, col + 1] != '*':
            possible_actions.append('right')
        return possible_actions

    def do_action(self, state: tuple, action: str) -> tuple:
        row, col = state
        new_states = {'up': (row - 1, col), 'down': (row + 1, col), 'left': (row, col - 1), 'right': (row, col + 1)}
        new_state = new_states[action]
        if not self.is_valid_state(new_state):
            new_state = state
        reward = self.board[new_state] if isinstance(self.board[new_state], int) else 0
        return reward, new_state

    def is_valid_state(self, state):
        row, col = state
        if row < 0 or row > self.dimensions[0]-1 or col < 0 or col > self.dimensions[1]-1 or self.board[state] == "*":
            return False
        return True

    def reset(self):
        self.current_state = (0, 0)

    def is_terminal(self, s = None) -> bool:
        if s:
            return s in self.terminals
        return self.current_state in self.terminals

    def set_traps(self, traps: list) -> None:
        for i, j in traps:
            self.board[i, j] = '*'

    def set_rewards(self, rewards: list) -> None:
        for i, j, v in rewards:
            self.board[i, j] = v
            # self.terminals.add((i, j))  # from what we've seen, all rewards are terminals. But it could be diff.

    def set_terminals(self, terminals: list) -> None:
        for tup in terminals:
            self.terminals.add(tup)

    def iterstates(self):
        for i in range(self.dimensions[0]):
            for j in range(self.dimensions[1]):
                if self.board[i, j] == "*":
                    continue
                yield (i, j)

    def show(self, d) -> None:  # just for UI purposes
        copy = self.board.copy()
        copy[self.current_state] = "ME"
        labels = np.arange(0, self.dimensions[1])
        print('  %s ' % (' '.join(f'%0{d}s' % i for i in labels)))
        print('  .%s.' % ('-'.join(f'%0{d}s' % f"{'-' * d}" for i in labels)))
        max_size = len(str(labels[-1]))
        for row_label, row in zip(labels, copy):
            row = [round(i, 2) if isinstance(i, float) else i for i in row]
            s = max_size - len(str(row_label))
            print(f'%s {" "*s}|%s|' % (row_label, '|'.join(f'%0{d}s' % i for i in row)))
            print(f'{" "*max_size} |%s|' % ('|'.join('%01s' % f"{'-' * d}" for i in row)))

class Qlearner:

    def __init__(self, env: Environment, initial_state: tuple, alpha: float = 0.9,
                 gamma: float = 0.9, epsilon: float = 0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_state = initial_state
        self.Q = self.init_Q()

    def init_Q(self) -> dict:
        q = {}
        for state in self.env.iterstates():
            for action in self.env.get_possible_actions(state):
                q[(state, action)] = 0
        return q

    def run(self, episodes: int):
        for episode in range(episodes):
            s = self.initial_state
            while not self.env.is_terminal(s):
                a = self.choose_action(s)
                r, s_ = self.env.do_action(s, a)
                self.Q[(s, a)] = self.action_function(s, a, r, s_)
                s = s_

    def choose_action(self, s: tuple) -> str:
        prob = random.uniform(0, 1)
        if prob <= self.epsilon:  # explore
            action = random.choice(self.env.get_possible_actions(s))
        else:  # exploit
            action = self.argmaxQ(s)
        return action

    def action_function(self, s: tuple, a: str, r: float, s_: str) -> float:
        return self.Q[(s, a)] + self.alpha * (r + self.gamma * self.maxQ(s_) - self.Q[(s, a)])

    def maxQ(self, s: tuple) -> float:
        return max(self.Q[(s, a)] for a in self.env.get_possible_actions(s))

    def argmaxQ(self, s: tuple) -> str:
        maxQ = self.maxQ(s)
        argmax = [a for a in self.env.get_possible_actions(s) if self.Q[(s, a)] == maxQ]
        return random.choice(argmax)

    ############################ METHODS THAT ARE NOT PART OF THE ALGORITHM #########################
    def get_steps(self):
        s, c = self.initial_state, 0
        visited = {s}
        while not self.env.is_terminal(s):
            a = self.argmaxQ(s)
            c += 1
            _, s = self.env.do_action(s, a)
            if s in visited:
                return -1
            visited.add(s)
        return c

    def show(self, actions=False, spacing=3) -> None:  # just for UI purposes
        actions_ui = {"right": "==>", "left": "<==", "up": "U", "down": "D"}
        d = spacing
        copy = self.env.board.copy()
        for state in self.env.iterstates():
            if self.env.is_terminal(state):
                continue
            policy = self.argmaxQ(state)
            if actions:
                action = actions_ui[policy]
            else:
                action = self.Q[(state, policy)]
            copy[state] = action
        labels = np.arange(0, self.env.dimensions[1])
        print('  %s ' % (' '.join(f'%0{d}s' % i for i in labels)))
        print('  .%s.' % ('-'.join(f'%0{d}s' % f"{'-' * d}" for i in labels)))
        max_size = len(str(labels[-1]))
        for row_label, row in zip(labels, copy):
            row = [round(i, 2) if isinstance(i, float) else i for i in row]
            s = max_size - len(str(row_label))
            print(f'%s {" " * s}|%s|' % (row_label, '|'.join(f'%0{d}s' % i for i in row)))
            print(f'{" " * max_size} |%s|' % ('|'.join('%01s' % f"{'-' * d}" for i in row)))


taxi_env = Environment((5,8))
traps = [(0,2), (1,2), (3,1), (4,1), (3,4), (4,4)]
taxi_env.set_traps(traps)
taxi_env.current_state = (1,6)
taxi_env.set_rewards([(4,5,1), (0,7, 5)])
taxi_env.set_terminals([(0,7)])

qlearner = Qlearner(taxi_env, (1,6), gamma=1, epsilon=0.1)
qlearner.run(10)
qlearner.show(actions=True, spacing=5)