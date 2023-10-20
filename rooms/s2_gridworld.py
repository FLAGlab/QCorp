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


def generate_grid_world_board() -> tuple[dict[Coordinates, float], Coordinates, Coordinates]:
    board = {}
    for x in range(10):
        for y in range(10):
            board[Coordinates(x, y)] = 0
    del board[Coordinates(1, 2)]
    del board[Coordinates(2, 2)]
    del board[Coordinates(3, 2)]
    del board[Coordinates(4, 2)]
    del board[Coordinates(6, 2)]
    del board[Coordinates(7, 2)]
    del board[Coordinates(8, 2)]
    del board[Coordinates(4, 3)]
    del board[Coordinates(4, 4)]
    del board[Coordinates(4, 5)]
    del board[Coordinates(4, 6)]
    del board[Coordinates(4, 7)]
    board[Coordinates(5, 4)] = -1
    board[Coordinates(5, 7)] = -1
    board[Coordinates(6, 7)] = -1
    return board, Coordinates(5, 5), Coordinates(0, 0)



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



def main():
    random_seed = 5657656
    grid_world_board, grid_world_objective, grid_world_start = generate_grid_world_board()
    grid_world = Grid(grid_world_board, grid_world_objective, grid_world_start)
    q_learning = QLearning(grid_world, random_seed, gamma=0.9)
    print(f"El agente toma {q_learning.run()} muestras para converger valores a 3 decimales")
