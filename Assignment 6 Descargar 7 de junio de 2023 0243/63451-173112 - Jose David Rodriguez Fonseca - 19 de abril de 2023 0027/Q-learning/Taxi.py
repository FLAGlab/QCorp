import random
from typing import List

import numpy as np


class TaxiWorld:
    def __init__(
        self,
        dimensions: tuple,
        unavailable_combinations: List[tuple[tuple, str]],
        transport_points: List[tuple],
        current_state: tuple = (0, 0),
        reward_value: float = 1.0,
        completed_travel_reward: float = 5.0,
        incompleted_travel_reward: float = -10.0,
        step_value: float = 0.0,
        noise_probs: dict = {},
    ):
        self.dimensions = dimensions
        self.board = np.zeros((self.dimensions), dtype="<U6")
        self.board[:, :] = str(step_value)

        self.incompleted_travel_reward = incompleted_travel_reward
        self.completed_travel_reward = completed_travel_reward
        self.current_state = current_state
        self.reward_value = str(reward_value)
        self.step_value = str(step_value)
        self.noise_probs = noise_probs
        self.unavailable_combinations = unavailable_combinations
        self.transport_points = transport_points
        self.set_passenger_features()
        self.passenger_actions = ["pick_up_passenger", "leave_passenger"]
        self.episode_completed = False

    def set_passenger_features(self):
        self.passenger = False
        self.passenger_position = random.choice(self.transport_points)
        self.passenger_destination = random.choice(
            [
                point
                for point in self.transport_points
                if point != self.passenger_position
            ]
        )
        self.episode_completed = False

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
            "leave_passenger": self.current_state,
            "pick_up_passenger": self.current_state,
        }
        possible_actions = [
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
            "leave_passenger",
            "pick_up_passenger",
        ]

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
        reward = (
            float(self.board[self.current_state])
            if action not in self.passenger_actions
            else self.calculate_travel_reward(action)  # type: ignore
        )

        return reward

    def calculate_travel_reward(self, action: str):
        if action == "leave_passenger":
            validated_action = (
                self.current_state == self.passenger_destination
                and self.passenger == True
            )
            if validated_action:
                self.episode_completed = True
                self.passenger = False
                return self.completed_travel_reward
            else:
                return self.incompleted_travel_reward

        elif action == "pick_up_passenger":
            validated_action = (
                self.current_state == self.passenger_position
                and self.passenger == False
            )
            if validated_action:
                self.passenger = True
                return float(self.reward_value)
            else:
                return self.incompleted_travel_reward

    def reset_state(self, new_initial_state: tuple = (0, 0)):
        self.current_state = new_initial_state
        self.set_passenger_features()

    def is_terminal(self):
        return self.episode_completed
