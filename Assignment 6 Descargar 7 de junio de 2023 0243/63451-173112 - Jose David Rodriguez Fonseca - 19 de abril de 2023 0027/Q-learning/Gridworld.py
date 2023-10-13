import random
from typing import List

import numpy as np


class Gridworld:
    def __init__(
        self,
        dimensions: tuple,
        trap_states: List[tuple[int]],
        disabled_states: List[tuple[int]],
        final_state: tuple,
        current_state: tuple = (0, 0),
        reward_value: float = 1.0,
        trap_value: float = -1.0,
        step_value: float = 0.0,
        noise_probs: dict = {},
    ):
        self.dimensions = dimensions
        self.board = np.zeros((self.dimensions), dtype="<U6")
        self.board[:, :] = str(step_value)
        self.trap_states = trap_states
        trap_rows, trap_cols = zip(*trap_states)
        self.board[trap_rows, trap_cols] = str(trap_value)
        dis_rows, dis_cols = zip(*disabled_states)
        self.board[dis_rows, dis_cols] = "*"
        self.board[final_state] = str(reward_value)
        if trap_value < -1:
            self.board[1, 0] = str(1.0)

        self.current_state = current_state
        self.reward_value = str(reward_value)
        self.trap_value = str(trap_value)
        self.step_value = str(step_value)

        self.final_state = final_state
        self.noise_probs = noise_probs

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
            if self.current_state not in self.trap_states
            and self.current_state != self.final_state
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
        self.current_state = (
            self.actions[action] if action is not None else self.current_state
        )
        reward = float(self.board[self.current_state])

        return reward

    def reset_state(self, new_initial_state: tuple = (0, 0)):
        self.current_state = new_initial_state

    def is_terminal(self, state=None):
        if state:
            return (
                self.board[state] != self.step_value
                and self.board[state] != "*"
            )
        return (
            True
            if self.board[self.current_state] != self.step_value
            and self.board[self.current_state] != "*"
            else False
        )
