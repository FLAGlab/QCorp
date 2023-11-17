#%pip install matplotlib pandas --q

from __future__ import annotations
from enum import Enum
from random import random, choice, seed
from math import ceil
from matplotlib.pyplot import Rectangle, subplots, rcParams
from pandas import DataFrame, set_option
from copy import deepcopy

#board
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



def generate_labyrinth() -> tuple[dict[Coordinates, float], list[tuple[Coordinates, Coordinates]], Coordinates, Coordinates]:
    board = {}
    for x in range(10):
        for y in range(10):
            board[Coordinates(x, y)] = 0
    blocked_paths = []
    for i in range(10):
        if i == 2 or i == 7:
            continue
        blocked_paths.append((Coordinates(i, 4), Coordinates(i, 5)))
        blocked_paths.append((Coordinates(4, i), Coordinates(5, i)))
    return board, blocked_paths, Coordinates(2, 0), Coordinates(6, 8)

set_option('display.max_rows', 500)
set_option('display.max_columns', 500)
set_option('display.width', 150)

rcParams['figure.figsize'] = [12, 8]
rcParams['figure.dpi'] = 100

#Q
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

#Rooms
def plot_scenario(scenario: QLearning):
    _, axes = subplots()
    axes.set_aspect('equal')
    axes.set_xlim(0, scenario.environment.dimensions.x)
    axes.set_ylim(scenario.environment.dimensions.y, 0)
    for x in range(scenario.environment.dimensions.x):
        for y in range(scenario.environment.dimensions.y):
            coordinates = Coordinates(x, y)
            if coordinates in scenario.environment.board:
                value = ceil((scenario.get_value(coordinates)) * 100) / 100
                color = 'white'
                qvalue = scenario.get_policy_qvalue(coordinates)
                if isinstance(scenario, QLearningTaxi):
                    if coordinates == scenario.environment.objective:
                        color = "green"
                    elif coordinates == scenario.environment.passenger:
                        color = "blue"
                    elif coordinates in scenario.environment.objectives:
                        color = "orange"
                else:
                    if coordinates == scenario.environment.objective:
                        color = "green"
                    elif coordinates == scenario.environment.start:
                        color = "blue"
                    elif value < 0:
                        color = "red"
                if qvalue != None:
                    qvalue = ceil(qvalue * 100) / 100
                    action = scenario.get_policy(coordinates)
                    action = "OUT" if action == None else action.name
                else:
                    qvalue = 0
                    action = "None"
                axes.add_patch(Rectangle((x, y), 1, 1, facecolor=color))
                if isinstance(scenario, QLearningTaxi):
                    text = "Con:{}\nQ:{:.2f}\nSin:{}\nQ:{:.2f}".format(scenario.get_policy_passenger(
                        coordinates).name, scenario.get_policy_qvalue_passenger(coordinates), action, qvalue)
                else:
                    text = "{}\nQ:{:.2f}\n{:.2f}".format(action, qvalue, value)
                axes.text(x + 0.5, y + 0.5, text, ha='center', va='center')
            else:
                axes.add_patch(Rectangle((x, y), 1, 1, facecolor='gray'))
    for (first, second) in scenario.environment.blocked_paths:
        if first.x == second.x:
            x = [first.x, first.x + 1]
            y = [second.y, second.y] if first.y < second.y else [first.y, first.y]
            axes.plot(x, y, color="black")
        elif first.y == second.y:
            y = [first.y, second.y + 1]
            x = [second.x, second.x] if first.x < second.x else [first.x, first.x]
            axes.plot(x, y, color="black")


def generate_q_table(scenario: QLearning) -> DataFrame:
    q_table = {"Estado": [], }
    for action in scenario.environment.get_actions():
        q_table[action.name] = []
    q_table["Objetivo"] = []
    for state in sorted(scenario.Q, key=lambda x: (x.x, x.y)):
        if len(scenario.Q[state]) < 2:
            continue
        q_table["Objetivo"].append(
            "Sí" if state == scenario.environment.objective else "No")
        q_table["Estado"].append(f"({state.x}, {state.y})")
        for action in scenario.Q[state]:
            q_table[action.name].append(scenario.Q[state][action])
    return DataFrame(q_table)


def main():
    random_seed = 5657656
    labyrinth_board, labyrinth_blocked_paths, labyrinth_objective, labyrinth_start = generate_labyrinth()
    labyrinth = GridBlockedPaths(labyrinth_board, labyrinth_objective, labyrinth_start, labyrinth_blocked_paths)
    q_learning = QLearning(labyrinth, random_seed, gamma=0.8)
    print(f"El agente toma {q_learning.run()} muestras para converger valores a 3 decimales")
    plot_scenario(q_learning)