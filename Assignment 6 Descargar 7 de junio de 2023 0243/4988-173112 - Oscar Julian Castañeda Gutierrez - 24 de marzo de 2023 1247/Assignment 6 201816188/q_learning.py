from board import Action, Coordinates, Grid, TaxiGrid, generate_taxi
from random import random, choice, seed
from copy import deepcopy
from enum import Enum


class QLearning:
    alpha: float
    environment: Grid
    epsilon: float
    gamma: float
    Q: dict[Coordinates, dict[Action, float]]
    policies: dict[Coordinates, tuple[Enum, float]]

    def __init__(self, environment: Grid, random_seed: int, alpha=0.3, epsilon=0.1, gamma=0.6) -> None:
        seed(random_seed)
        self.alpha = alpha
        self.environment = environment
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = {}
        self.Q[self.environment.end_state] = {
            self.environment.end_action: 0}
        self.policies = {}

    def action_function(self, state: Coordinates, action: Action, reward: float, next_state: Coordinates, next_action: Action) -> float:
        q_value = self.get_qvalue(state, action)
        return q_value + self.alpha * (reward + self.gamma * self.get_qvalue(next_state, next_action) - q_value)

    def choose_action(self, state: Coordinates) -> Action:
        if state not in self.Q or len(self.Q[state]) < 1:
            return choice(self.environment.get_actions())
        return max(self.Q[state], key=self.Q[state].get)

    def get_qvalue(self, state: Coordinates, action: Action) -> float:
        if state not in self.Q:
            self.Q[state] = {action: 0}
        elif action not in self.Q[state]:
            self.Q[state][action] = 0
        return self.Q[state][action]

    def run(self) -> int:
        converges = False
        iteration = 0
        available_states = list(self.environment.board.keys())
        while not converges:
            self.environment.reset()
            converges = True
            iteration += 1
            state = choice(available_states)
            Qs = deepcopy(self.Q)
            steps = 0
            while True:
                steps +=1
                if steps > 200:
                    break
                if random() < self.epsilon:
                    action = choice(self.environment.get_actions())
                else:
                    action = self.choose_action(state)
                reward, next_state = self.environment.do_action(state, action)
                next_action = self.environment.end_action if next_state == self.environment.end_state else self.choose_action(
                    next_state)
                self.Q[state][action] = self.action_function(
                    state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                if action == self.environment.end_action:
                    break
            if steps > 200:
                converges = False
                self.Q = Qs
                continue
            for coordinates in self.environment.board:
                if coordinates not in Qs:
                    converges = False
                    continue
                for action in self.environment.get_actions():
                    if action not in Qs[coordinates] or abs(self.Q[coordinates][action] - Qs[coordinates][action]) > 0.001:
                        converges = False
        for state in self.Q:
            action = max(self.Q[state], key=self.Q[state].get)
            self.policies[state] = (action, self.Q[state][action])
        return iteration

    def get_policy_qvalue(self, state: Coordinates) -> float:
        return self.policies[state][1]

    def get_value(self, state: Coordinates) -> float:
        return self.environment.board[state]

    def get_policy(self, state: Coordinates) -> Enum:
        return self.policies[state][0]


class QLearningTaxi(QLearning):
    environment: TaxiGrid
    policies_passenger: dict[Coordinates, tuple[Enum, float]]
    Q: dict[tuple[Coordinates, bool], dict[Action, float]]

    def __init__(self, environment: Grid, random_seed: int, alpha=0.3, epsilon=0.1, gamma=0.6) -> None:
        super().__init__(environment, random_seed, alpha, epsilon, gamma)
        self.policies_passenger = {}
        del self.Q[self.environment.end_state]
        self.Q[(self.environment.end_state, True)] = {
            self.environment.end_action: 0}

    def run(self) -> int:
        converges = False
        iteration = 0
        available_states = list(self.environment.board.keys())
        while not converges:
            self.environment.reset()
            converges = True
            iteration += 1
            picked_passenger = random() < 0.5
            state = (choice(available_states), picked_passenger)
            self.environment.start = state[0]
            self.environment.picked_passenger = picked_passenger
            Qs = deepcopy(self.Q)
            steps = 0
            while True:
                steps +=1
                if steps > 200:
                    break
                if random() < self.epsilon:
                    action = choice(self.environment.get_actions())
                else:
                    action = self.choose_action(state)
                reward, next_state = self.environment.do_action(state[0], action)
                next_state = (next_state, self.environment.picked_passenger)
                next_action = self.environment.end_action if next_state[0] == self.environment.end_state else self.choose_action(next_state)
                self.Q[state][action] = self.action_function(state, action, reward, next_state, next_action)
                state, action = next_state, next_action
                if action == self.environment.end_action:
                    break
            if steps > 200:
                converges = False
                self.Q = Qs
                continue
            for coordinates in self.environment.board:
                for passenger in [False, True]:
                    q = (coordinates, passenger)
                    if q not in Qs:
                        converges = False
                        continue
                    for action in self.environment.get_actions():
                        if action not in Qs[q] or abs(self.Q[q][action] - Qs[q][action]) > 0.001:
                            converges = False
        for state in self.Q:
            action = max(self.Q[state], key=self.Q[state].get)
            policies = self.policies if not state[1] else self.policies_passenger
            if state[0] not in policies or policies[state[0]][1] < self.Q[state][action]:
                policies[state[0]] = (action, self.Q[state][action])
        return iteration

    def get_policy_qvalue_passenger(self, state: Coordinates) -> float:
        return self.policies_passenger[state][1]

    def get_policy_passenger(self, state: Coordinates) -> Enum:
        return self.policies_passenger[state][0]
