import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Gridworld import Gridworld
from Rooms_labyrinth import RoomsLabyrinth
from Taxi import TaxiWorld


class QLearningAgent:
    def __init__(
        self,
        world_params: dict,
    ):
        """
        Initialize the env features and define the hyperparameters of the model

        Args:
            world_params (dict): Dictionary that has the configuration for the
            given world
        """
        self.world_params = world_params
        self.gamma = self.world_params["gamma"]
        self.alpha = self.world_params["alpha"]
        self.epsilon = self.world_params["epsilon"]
        self.actions = self.world_params["actions"]
        self.disabled_states = []
        self.step_value = self.world_params["step_value"]
        self.Q = {}
        self.define_env()
        # Initialize Q values = 0
        if self.world_params["env_name"] != "taxi":
            for i in range(self.env.dimensions[0]):
                for j in range(self.env.dimensions[1]):
                    self.env.set_current_state((i, j))
                    if (
                        not self.env.is_terminal()
                        and self.env.current_state not in self.disabled_states
                    ):
                        self.Q[(i, j)] = {}
                        for a in self.actions:
                            self.Q[(i, j)][a] = 0.0
                    elif self.env.is_terminal():
                        self.Q[i, j] = {}
                        for a in world_params["terminal_actions"]:
                            self.Q[(i, j)][a] = 0.0
        else:
            for i in range(self.env.dimensions[0]):
                for j in range(self.env.dimensions[1]):
                    self.env.set_current_state((i, j))
                    if self.env.current_state not in self.disabled_states:
                        self.Q[(i, j)] = {}
                        for a in self.actions:
                            self.Q[(i, j)][a] = 0.0

        self.num_episodes = world_params["num_episodes"]
        self.policy_results = np.copy(self.env.board)
        self.policy_dict = {
            "up": "↑",
            "down": "↓",
            "left": "←",
            "right": "→",
            "salir": "-",
            "pick_up_passenger": "in",
            "leave_passenger": "out",
        }

    def define_env(self):
        """
        Initialize the environment the agent will interact with
        """
        if self.world_params["env_name"] == "Gridworld":
            self.trap_states = self.world_params["trap_states"]
            self.final_state = self.world_params["final_state"]
            self.disabled_states = self.world_params["disabled_states"]
            self.env = Gridworld(
                dimensions=self.world_params["dimensions"],
                trap_states=self.trap_states,
                disabled_states=self.disabled_states,
                final_state=self.final_state,
                reward_value=self.world_params["reward_value"],
                trap_value=self.world_params["trap_value"],
                step_value=self.world_params["step_value"],
            )
        elif self.world_params["env_name"] == "labyrinth":
            self.final_state = self.world_params["final_state"]
            self.env = RoomsLabyrinth(
                dimensions=self.world_params["dimensions"],
                unavailable_combinations=self.world_params[
                    "unavailable_combinations"
                ],
                final_state=self.final_state,
                reward_value=self.world_params["reward_value"],
                step_value=self.world_params["step_value"],
            )
        elif self.world_params["env_name"] == "taxi":
            self.env = TaxiWorld(
                transport_points=self.world_params["transport_points"],
                dimensions=self.world_params["dimensions"],
                unavailable_combinations=self.world_params[
                    "unavailable_combinations"
                ],
                reward_value=self.world_params["reward_value"],
                step_value=self.world_params["step_value"],
            )

    def run_episode(self):
        game_over = False
        initial_state = (
            (0, 0)
            if self.world_params["env_name"] == "Gridworld"
            else random.choice(list(self.Q.keys()))
        )
        self.env.reset_state(initial_state)
        s = self.env.current_state
        steps = 0
        while not game_over:
            action = self.get_action()
            self.Q[s][action] += self.get_qvalue(action)

            if self.env.is_terminal() or steps == 1000:
                game_over = True
            s = self.env.current_state
            steps += 1

    def run_Qlearning_algorithm(self):
        """
        Execute the Q-learnign algorithm for the defined environment.
        """
        for _ in range(self.num_episodes):
            self.run_episode()
        print(self.Q)
        self.plot_policy(self.num_episodes)
        self.generate_Q_table()

    def compute_qvalue(self, action: str):
        s = self.env.current_state
        reward = self.env.do_action(action)
        if self.world_params["env_name"] == "taxi":
            if self.env.episode_completed:  # type: ignore
                return self.alpha * (
                    reward + self.gamma * 0 - self.Q[s][action]
                )

        a_prime = self.compute_action_from_current_state(True)
        return self.alpha * (
            reward
            + self.gamma * self.Q[self.env.current_state][a_prime]
            - self.Q[s][action]
        )

    def compute_action_from_current_state(self, get_max: bool = False):
        probability = random.random()
        if probability < (1 - self.epsilon) or get_max:
            max_value = max(self.Q[self.env.current_state].values())
            max_keys = [
                key
                for key, val in self.Q[self.env.current_state].items()
                if val == max_value
            ]
            action = random.choice(max_keys)

            return action

        else:
            return random.choice(list(self.Q[self.env.current_state].keys()))

    def get_action(self):
        return self.compute_action_from_current_state(get_max=False)

    def get_qvalue(self, action):
        return self.compute_qvalue(action)

    def plot_policy(self, iterations: int):
        for state in self.Q.keys():
            self.env.reset_state(state)
            if self.env.board[state] != self.step_value and self.env.board[
                state
            ] != str(self.step_value):
                self.policy_results[state] = "-"
            else:
                self.policy_results[state] = self.policy_dict[
                    self.compute_action_from_current_state(True)
                ]

        figsize = (10, 10)
        title = f"results for {iterations} iterations"
        _, matrix = np.unique(self.policy_results, return_inverse=True)

        plt.figure(figsize=figsize)
        plt.title(title)
        ax = sns.heatmap(
            matrix.reshape(self.policy_results.shape),
            annot=self.policy_results,
            annot_kws={"fontsize": 25},
            fmt="",
            cmap=sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True),
        )
        ax.set(xticklabels=[], yticklabels=[])
        ax.tick_params(bottom=False, left=False)
        plt.savefig(
            f'results/{self.world_params["env_name"]}_policy_{iterations}_'
            "iterations.jpg"
        )

    def generate_Q_table(self):
        Q_table = pd.DataFrame(self.Q)
        Q_table.to_excel(f'results/{self.world_params["env_name"]}.xlsx')


if __name__ == "__main__":
    gridworld_params = {
        "env_name": "Gridworld",
        "dimensions": (10, 10),
        "disabled_states": [
            (2, 1),
            (2, 2),
            (2, 3),
            (2, 4),
            (2, 6),
            (2, 7),
            (2, 8),
            (3, 4),
            (4, 4),
            (5, 4),
            (6, 4),
            (7, 4),
        ],
        "trap_states": [(4, 5), (7, 5), (7, 6)],
        "final_state": (5, 5),
        "reward_value": 1.0,
        "trap_value": -1.0,
        "step_value": 0.0,
        "gamma": 0.9,
        "alpha": 0.5,
        "epsilon": 0.9,
        "num_episodes": 10000,
        "actions": ["up", "down", "left", "right"],
        "terminal_actions": ["salir"],
    }
    unavailable_labyrinth_states = (
        [((4, idx), "down") for idx in range(0, 10) if idx not in [2, 7]]
        + [((5, idx), "up") for idx in range(0, 10) if idx not in [2, 7]]
        + [((idx, 4), "right") for idx in range(0, 10) if idx not in [2, 7]]
        + [((idx, 4), "left") for idx in range(0, 10) if idx not in [2, 7]]
    )

    labyrinth_params = {
        "env_name": "labyrinth",
        "dimensions": (10, 10),
        "unavailable_combinations": unavailable_labyrinth_states,
        "final_state": (0, 2),
        "reward_value": 1.0,
        "step_value": 0.0,
        "gamma": 0.9,
        "alpha": 0.5,
        "epsilon": 0.8,
        "num_episodes": 10000,
        "actions": ["up", "down", "left", "right"],
        "terminal_actions": ["salir"],
    }

    unavailable_taxi_states = [
        ((3, 0), "right"),
        ((4, 0), "right"),
        ((3, 1), "left"),
        ((4, 1), "left"),
        ((0, 1), "right"),
        ((1, 1), "right"),
        ((0, 2), "left"),
        ((1, 2), "left"),
        ((3, 2), "right"),
        ((4, 2), "right"),
        ((3, 3), "left"),
        ((4, 3), "left"),
    ]
    taxi_params = {
        "env_name": "taxi",
        "dimensions": (5, 5),
        "unavailable_combinations": unavailable_taxi_states,
        "transport_points": [(0, 0), (4, 0), (0, 4), (4, 3)],
        "reward_value": 1.0,
        "step_value": 0.0,
        "gamma": 0.9,
        "alpha": 0.5,
        "epsilon": 0.9,
        "num_episodes": 1000,
        "actions": [
            "up",
            "down",
            "left",
            "right",
            "leave_passenger",
            "pick_up_passenger",
        ],
    }
    QLearningAgent(gridworld_params).run_Qlearning_algorithm()
    QLearningAgent(labyrinth_params).run_Qlearning_algorithm()
    QLearningAgent(taxi_params).run_Qlearning_algorithm()
