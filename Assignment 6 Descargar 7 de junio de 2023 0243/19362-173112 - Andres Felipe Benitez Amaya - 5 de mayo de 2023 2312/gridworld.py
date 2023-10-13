import random
import numpy as np 

class Gridworld:
    def __init__(self, n=5):
        self.dimensions = (n, n)
        self.board = [[' ' for _ in range(n)] for _ in range(n)]
        self.board[0][0] = 'S' 
        self.board[-1][-1] = 'G' 
        self.board[-2][-2] = 'L' 
        self.board[-1][-2] = 'X' 
        self.board[-2][-1] = 'X' 
        self.current_state = (0, 0)
        self.goal_state = [(n-1, n-1), (n-2, n-2)]
        self.trap_state = [(n-1, n-2), (n-2, n-1)]
        self.rewards = {(i, j): 0 for i in range(n) for j in range(n)}
        self.rewards[self.goal_state[0]] = 1
        self.rewards[self.goal_state[1]] = -1
        self.rewards[self.trap_state[0]] = -1
        self.rewards[self.trap_state[1]] = -1

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        i, j = state
        possible_actions = []
        if i > 0 and self.board[i-1][j] != '*':
            possible_actions.append('up')
        if i < self.dimensions[0]-1 and self.board[i+1][j] != '*':
            possible_actions.append('down')
        if j > 0 and self.board[i][j-1] != '*':
            possible_actions.append('left')
        if j < self.dimensions[1]-1 and self.board[i][j+1] != '*':
            possible_actions.append('right')
        return possible_actions

    def do_action(self, action):
        i, j = self.current_state
        if action == 'up':
            new_state = (i-1, j)
        elif action == 'down':
            new_state = (i+1, j)
        elif action == 'left':
            new_state = (i, j-1)
        elif action == 'right':
            new_state = (i, j+1)
        reward = self.rewards[new_state]
        self.current_state = new_state
        return reward, new_state

    def reset(self):
        self.current_state = (0, 0)

    def is_terminal(self):
        return self.current_state in self.goal_state
    
    def q_learning(env, alpha=0.1, gamma=0.9, epsilon=0.1, num_episodes=10000):
        q_table = {(i, j): {'up': 0, 'down': 0, 'left': 0, 'right': 0} for i in range(env.dimensions[0]) for j in range(env.dimensions[1])}
        for episode in range(num_episodes):
            env.reset()
            done = False
            while not done:
                state = env.get_current_state()
                if random.uniform(0, 1) < epsilon:
                    action = random.choice(env.get_possible_actions(state))

        # Ejecutar el algoritmo 
        num_episodes = 1000
        for episode in range(num_episodes):
            state = env.reset()
            done = False

            while not done:
                action = epsilon_greedy(q_table[state[0], state[1]], episode)
                reward, new_state = env.do_action(action)
                q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] + alpha * (reward + gamma * np.max(q_table[new_state[0], new_state[1]]))
                state = new_state
                done = env.is_terminal()

        policy = np.argmax(q_table, axis=2)
        total_reward = 0
        state = env.reset()
        done = False

        while not done:
            action = policy[state[0], state[1]]
            reward, new_state = env.do_action(action)
            total_reward += reward
            state = new_state
            done = env.is_terminal()

        print("Politica resultado:")
        print(policy)
        print("Recompensa total:", total_reward)

        # Imprimir la Q-tabla
        print("Q-tabla:")
        print(q_table)