from environment import Environment
import random
import numpy as np


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
