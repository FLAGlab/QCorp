import random
import numpy as np
import pandas as pd

class Envirionment:
    def __init__(self, board, dimensions):
        self.board = board
        self.dimensions = dimensions
        self.current_state = self.initialize_state()
        self.actions = ['up', 'down', 'left', 'right']

    def initialize_state(self):
        n = True
        while n == True:
            i = random.randint(0,self.dimensions[0]-1)
            j = random.randint(0,self.dimensions[1]-1)
            if self.board[i][j] != "*" or self.board[i][j] != "1":
                n = False
        return (i,j)
        
    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        i, j = state
        actions = ['up', 'down', 'left', 'right']
        if i == 0 or self.board[max(i-1,0)][max(j,0)]=="*":
            actions.remove('up')
        if i == self.dimensions[0] - 1 or self.board[min(i+1,self.dimensions[0] - 1)][max(j,0)]=="*":
            actions.remove('down')
        if j == 0 or self.board[max(i,0)][max(j-1,0)]=="*":
            actions.remove('left')
        if j == self.dimensions[1] - 1 or self.board[max(i,0)][min(j+1,self.dimensions[1] - 1)]=="*":
            actions.remove('right')
        return actions

    def do_action(self, action):
        i, j = self.current_state
        if action == 'up':
            i = max(i - 1, 0)
        elif action == 'down':
            i = min(i + 1, self.dimensions[0] - 1)
        elif action == 'left':
            j = max(j - 1, 0)
        elif action == 'right':
            j = min(j + 1, self.dimensions[1] - 1)
        self.current_state = (i, j)
        
        if self.board[i][j] == " ":
            reward = 0
        elif self.board[i][j] == "1":
            reward = 1
        elif self.board[i][j] == "-1":
            reward = -1
        return reward, self.current_state

    def reset(self):
        self.current_state = (0, 0)

    def is_terminal(self, state):
        return self.board[state[0]][state[1]] == "1"
    

board = [[" "," ","1"," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," "," "," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         ["*","*"," ","*","*","*","*","*"," ","*","*"],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," "," "," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "],
         [" "," "," "," "," ","*"," "," "," "," "," "]]
dimensions = (11, 11)
env = Envirionment(board, dimensions)


class q_learning:
    
    def __init__(self, env, gamma, alpha, epsilon):
        self.env = env
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.Q = {}
        
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            act = self.env.get_possible_actions(state)
            action = random.choice(act)
        else:
            action = self.max_action(state)
        return action
    
    def max_action(self, state):
        max_val = None
        for key, val in self.Q.items():
            if key[0] == state:
                if max_val is None or val > max_val:
                    max_val = val
                    action = key[1]
        return action
    
    def action_function(self, state1, action1, reward, state2, max_act):
        q_value = (1- self.alpha)* self.Q[state1,action1] + self.alpha*(float(reward) + self.gamma*self.Q[state2,max_act])
        self.Q[state1,action1] = q_value

        
    def init_q(self):
        for i in range (0,self.env.dimensions[0]):
            for j in range (0,self.env.dimensions[1]):
                actions = self.env.get_possible_actions((i,j))
                for n in actions:
                    self.Q[(i,j),n] = 0
    
    def train(self, episodes):
        self.init_q()
        for i in range (0,episodes):
            state = self.env.current_state
            action = self.choose_action(state)
            while not self.env.is_terminal(state):
                reward, state2 =  self.env.do_action(action)
                max_act = self.max_action(state2)
                self.action_function(state, action, reward, state2, max_act)
                state = state2
                action2 = self.choose_action(state2)
                action = action2
            self.env.reset()
            
    def show(self, board):
        valor = {}
        politica = {}
        for key, val in self.Q.items():
            if key[0] not in valor:
                valor[key[0]] = val
                politica[key[0]] = key[1]
            else:
                if val > valor[key[0]]:
                    valor[key[0]] = val
                    politica[key[0]] = key[1]
        for n in politica.keys():
            i,j = n
            if board[i][j]== " ":
                board[i][j] = politica[n]
            df = pd.DataFrame(board)
        return df
q_model = q_learning(env,gamma = 0.81, alpha = 0.96, epsilon = 0.9)


def main():
    q_model.init_q()   
    q_model.train(5000)
    q_model.show(board) 