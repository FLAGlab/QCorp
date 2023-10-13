import random
from typing import List

import numpy as np


class RoomsLabyrinth:
    def __init__(
        self,
        dimensions: tuple,
        unavailable_combinations: List[tuple[tuple, str]],
        final_state: tuple,
        current_state: tuple = (0, 0),
        reward_value: float = 1.0,
        step_value: float = 0.0,
        noise_probs: dict = {},
    ):
        self.dimensions = dimensions
        self.board = np.zeros((self.dimensions), dtype="<U6")
        self.board[:, :] = str(step_value)
        self.board[final_state] = str(reward_value)

        self.current_state = current_state
        self.reward_value = str(reward_value)
        self.step_value = str(step_value)

        self.final_state = final_state
        self.noise_probs = noise_probs
        self.unavailable_combinations = unavailable_combinations

    def get_current_state(self):
        return self.current_state

    def set_current_state(self, state):
        self.current_state = state

    def get_possible_actions(
        self,
    ):
        self.actions = {
            "up": (self.current_state[0] - 1, self.current_state[1]),
            "down": (self.current_state[0] + 1, self.current_state[1]),
            "left": (self.current_state[0], self.current_state[1] - 1),
            "right": (self.current_state[0], self.current_state[1] + 1),
            "salir": self.current_state,
        }
        possible_actions = (
            [
                "up"
                if self.actions["up"][0] >= 0
                and self.board[self.actions["up"]] != "*"
                else None,
                "down"
                if self.actions["down"][0] < self.dimensions[0]
                and self.board[self.actions["down"]] != "*"
                else None,
                "left"
                if self.actions["left"][1] >= 0
                and self.board[self.actions["left"]] != "*"
                else None,
                "right"
                if self.actions["right"][1] < self.dimensions[1]
                and self.board[self.actions["right"]] != "*"
                else None,
            ]
            if self.current_state != self.final_state
            else ["salir"]
        )

        return possible_actions

    def do_action(self, action="right"):
        possible_actions = self.get_possible_actions()

        if self.noise_probs:
            if random.random() < self.noise_probs[action]:
                action = random.choice(possible_actions)  # type: ignore

        if action not in possible_actions:
            action = None

        move_condition = (
            action is not None
            and (self.current_state, action)
            not in self.unavailable_combinations
        )
        self.current_state = (
            self.actions[action] if move_condition else self.current_state  # type: ignore
        )
        reward = float(self.board[self.current_state])

        return reward

    def reset_state(self, new_initial_state: tuple = (0, 0)):
        self.current_state = new_initial_state

    def is_terminal(self, state=None):
        if state:
            return self.board[state] != self.step_value
        return (
            True
            if self.board[self.current_state] != self.step_value
            else False
        )
