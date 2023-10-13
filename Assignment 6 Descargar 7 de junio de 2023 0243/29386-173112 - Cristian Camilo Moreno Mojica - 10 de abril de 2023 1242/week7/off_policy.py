import numpy as np


class OffPolicyControl:

    def __init__(self, env: object, epsilon: float, Q: np.array, alpha: float, gamma: float,
                 reward_modified: dict = None):
        self.env = env
        self.epsilon = epsilon
        self.Q = Q
        self.alpha = alpha
        self.gamma = gamma
        self.reward_modified = reward_modified
        self.number_actions = Q.shape[1]
        self.reward_history = {}
        self.actions_dict = {0: 'rigth', 1: 'left', 2: 'up', 3: 'down', 4: 'go off'}

    def action_epsilon_greedy(self, state: int):
        if np.random.rand() > self.epsilon:
            return np.argmax(self.Q[state])
        return np.random.randint(self.number_actions)

    def update_value(self, state: int, action: int, reward: int, new_state: int, action_max: int):
        if self.reward_modified is None:
            reward = reward
        else:
            reward = self.reward_modified[reward]
        self.Q[state, action] = self.Q[state, action] + self.alpha * (
                reward + self.gamma * self.Q[new_state, action_max] - self.Q[state, action])

    def episode(self, episode_value: int):
        done = False
        state = self.env.reset()
        g = 0
        while not done:
            action = self.action_epsilon_greedy(state)
            new_state, reward, done, info = self.env.step(action)
            action_max = np.argmax(self.Q[new_state])
            self.update_value(state, action, reward, new_state, action_max)
            state = new_state
            g = + reward
            if episode_value % 10:
                self.reward_history[episode_value] = g

    def iter_episode(self, episodes: int):
        for i in range(episodes):
            self.episode(i)

    def select_polices(self):
        result = []
        list_board = self.env.board.tolist()
        flat_list = [item for sublist in list_board for item in sublist]
        for i in range(0, self.Q.shape[0]):
            if flat_list[i] == ' ':
                value = self.Q[i]
                value[value == 0] = -np.inf
                result.append(self.actions_dict[np.argmax(value)])
            else:
                result.append('*')
        return np.array(result).reshape(self.env.dimensions)