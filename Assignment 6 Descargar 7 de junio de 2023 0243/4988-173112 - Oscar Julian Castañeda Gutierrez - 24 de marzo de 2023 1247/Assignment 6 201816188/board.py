from __future__ import annotations
from enum import Enum
from random import choice


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
