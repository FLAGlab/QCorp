from __future__ import annotations
from random import random, choice, seed
from copy import deepcopy
from enum import Enum


class Coordinates:
    x: int
    y: int

    def __init__(self, x: int, y: int) -> None:
        self.x = x
        self.y = y

    def __add__(self, other: Coordinates) -> Coordinates:
        return Coordinates(self.x + other.x, self.y + other.y)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: Coordinates) -> bool:
        return isinstance(other, Coordinates) and (self.x, self.y) == (other.x, other.y)

    def __str__(self) -> str:
        return f"<Coordinate x: {self.x} y: {self.y}>"


class Action(Enum):
    DOWN = Coordinates(0, 1)
    LEFT = Coordinates(-1, 0)
    RIGHT = Coordinates(1, 0)
    UP = Coordinates(0, -1)
    OUT = Coordinates(0, 0)


class Grid():
    blocked_paths: list[tuple[Coordinates, Coordinates]]
    board: dict[Coordinates, float]
    dimensions: Coordinates
    end_action: Action
    end_state: Coordinates
    objective: Coordinates
    objective_reward: float
    start: Coordinates

    def __init__(self, board: dict[Coordinates, float], objective: Coordinates, start: Coordinates, blocked_paths=[]) -> None:
        self.blocked_paths = blocked_paths
        self.board = board
        self.dimensions = Coordinates(0, 0)
        for coordinate in board:
            if coordinate.x + 1 > self.dimensions.x:
                self.dimensions.x = coordinate.x + 1
            if coordinate.y + 1 > self.dimensions.y:
                self.dimensions.y = coordinate.y + 1
        self.end_action = Action.OUT
        self.end_state = Coordinates(-1, -1)
        self.objective_reward = 1
        self.objective = objective
        self.start = start

    def get_actions(self) -> list[Action]:
        return list(Action)

    def reset(self):
        return

    def do_action(self, coordinates: Coordinates, action: Action) -> tuple[float, Coordinates]:
        new_coordinates: Coordinates = coordinates + action.value
        if action == Action.OUT:
            if coordinates == self.objective:
                return self.objective_reward, self.end_state
            return 0, coordinates
        if new_coordinates not in self.board:
            new_coordinates = coordinates
        return self.board[new_coordinates], new_coordinates


class GridBlockedPaths(Grid):

    def __init__(self, board: dict[Coordinates, float], objective: Coordinates, start: Coordinates, blocked_paths=[]) -> None:
        super().__init__(board, objective, start, blocked_paths)

    def do_action(self, coordinates: Coordinates, action: Action) -> tuple[float, Coordinates]:
        new_coordinates: Coordinates = coordinates + action.value
        if action == Action.OUT:
            if coordinates == self.objective:
                return self.objective_reward, self.end_state
            return 0, coordinates
        if new_coordinates not in self.board or (coordinates, new_coordinates) in self.blocked_paths or (new_coordinates, coordinates) in self.blocked_paths:
            new_coordinates = coordinates
        return self.board[new_coordinates], new_coordinates


class TaxiAction(Enum):
    DOWN = Coordinates(0, 1)
    LEFT = Coordinates(-1, 0)
    RIGHT = Coordinates(1, 0)
    UP = Coordinates(0, -1)
    PICK = 0
    DROP = 1


class TaxiGrid(GridBlockedPaths):
    end_action: TaxiAction
    picked_passenger: bool
    passenger: Coordinates
    objectives: list[Coordinates]
    pick_reward: float
    drop_reward: float
    drop_penalty: float
    pick_penalty: float

    def __init__(self, board: dict[Coordinates, float], objectives: list[Coordinates], start: Coordinates, blocked_paths=[]) -> None:
        super().__init__(board, Coordinates(-1, -1), start, blocked_paths)
        self.end_action = TaxiAction.DROP
        self.drop_penalty = -10
        self.pick_penalty = -10
        self.drop_reward = 5
        self.pick_reward = 1
        self.objectives = objectives

    def get_actions(self) -> list[TaxiAction]:
        return list(TaxiAction)

    def reset(self):
        self.picked_passenger = False
        self.passenger = choice(self.objectives)
        self.objective = choice(
            list(filter(lambda objective: objective != self.passenger, self.objectives)))

    def do_action(self, coordinates: Coordinates, action: TaxiAction) -> tuple[float, Coordinates]:
        if action == TaxiAction.DROP:
            if coordinates == self.objective and self.picked_passenger:
                return self.drop_reward, self.end_state
            return self.drop_penalty, coordinates
        if action == TaxiAction.PICK:
            if coordinates == self.passenger and not self.picked_passenger:
                self.picked_passenger = True
                return self.pick_reward, coordinates
            return self.pick_penalty, coordinates
        new_coordinates: Coordinates = coordinates + action.value
        if new_coordinates not in self.board or (coordinates, new_coordinates) in self.blocked_paths or (new_coordinates, coordinates) in self.blocked_paths:
            new_coordinates = coordinates
        return self.board[new_coordinates], new_coordinates

def generate_taxi() -> tuple[dict[Coordinates, float], list[tuple[Coordinates, Coordinates]], list[Coordinates], Coordinates]:
    board = {}
    for x in range(5):
        for y in range(5):
            board[Coordinates(x, y)] = 0
    blocked_paths = [
        (Coordinates(0, 3), Coordinates(1, 3)),
        (Coordinates(0, 4), Coordinates(1, 4)),
        (Coordinates(1, 0), Coordinates(2, 0)),
        (Coordinates(1, 1), Coordinates(2, 1)),
        (Coordinates(2, 3), Coordinates(3, 3)),
        (Coordinates(2, 4), Coordinates(3, 4)),
    ]
    objectives = [
        Coordinates(0, 0),
        Coordinates(0, 4),
        Coordinates(4, 0),
        Coordinates(3, 4)
    ]
    return board, blocked_paths, objectives, Coordinates(3, 1)

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
    

def main():
    random_seed = 5657656
    taxi_board, taxi_blocked_paths, taxi_objective, taxi_start = generate_taxi()
    taxi = TaxiGrid(taxi_board, taxi_objective, taxi_start, taxi_blocked_paths)
    q_learning = QLearningTaxi(taxi, random_seed, alpha=0.1, epsilon=0.3)
    print(f"El agente toma {q_learning.run()} muestras para converger valores a 3 decimales")
    #plot_scenario(q_learning)