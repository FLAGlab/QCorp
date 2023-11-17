def show_values(board, decimals = 2):
    for i in range(0, len(board)):
        print('--------------------------------------------------------------------------------------------------')
        out = '| '
        for j in range(0, len(board[0])):
            value = 0
            if type(board[i][j]) == str:
                value = board[i][j]
            else:
                value = round(board[i][j],decimals)
            out += str(value).ljust(7) + ' | '
        print(out)
    print('--------------------------------------------------------------------------------------------------')

import numpy as np
import copy
import random

class Gridworld:
    #Dimensions[rows,cols]
    #Each cell element [row,col,value]
    def __init__(self, dimensions=[10,10], cells={}, initial_state=None, actions=['up','down','left','right']):
        self.dimensions = dimensions
        self.actions = actions
        self.cells = cells
        self.state_actions = self.define_actions()
        self.init_state(initial_state)
        
    def define_actions(self):
        state_actions = {}
        for row in range(self.dimensions[0]):
            for col in range(self.dimensions[1]):
                state = (row, col)
                if state in self.cells:
                    actions = self.actions.copy()
                    if 'e' in self.cells[state]:
                        actions.append('exit')
                    if 'l' in self.cells[state]:
                        actions.remove('left')
                    if 't' in self.cells[state]:
                        actions.remove('up')
                    if 'r' in self.cells[state]:
                        actions.remove('right')
                    if 'b' in self.cells[state]:
                        actions.remove('down')
                    state_actions[state] = actions
                else:
                    state_actions[state] = self.actions 
        return state_actions
        
    def get_board(self):
        return self.board
        
    def get_current_state(self):
        return self.current_state
    
    def get_possible_actions(self, state=(0,0)):
        return self.state_actions[state]
    
    def do_action(self, action, state = None):
        current_state = self.current_state
        reward = 0
        if state != None:
            current_state = state
        new_state = ()
        if(action=='up'):
            new_state=(current_state[0]-1,current_state[1])
        elif(action=='down'):
            new_state=(current_state[0]+1,current_state[1])
        elif(action=='left'):
            new_state=(current_state[0],current_state[1]-1)
        elif(action=='right'):
            new_state=(current_state[0],current_state[1]+1)
        elif(action=='exit'):
            new_state=(current_state[0],current_state[1])
            return [100, new_state]
        
        return [-1, new_state]
    
    def init_state(self, initial_state=None):
        if initial_state:
            self.initial_state = initial_state
            self.current_state = initial_state
        else:
            new_initial_state = random.choice(list(self.state_actions.keys()))
            self.initial_state = new_initial_state
            self.current_state = new_initial_state
        
    def is_terminal(self, action):
        if action == 'exit':
            return True
        else:
            return False


class QLearning:
    #Dimensions[rows,cols]
    #Each cell element [row,col,value]
    def __init__(self, mdp, discount=0.9, alpha=0.5, iterations=False, epsilon=0.9):
        # Mdp is equivalent to env
        self.mdp = mdp
        self.alpha = alpha
        self.discount = discount
        self.iterations = iterations
        self.epsilon = epsilon
        self.q = {}
        state_actions = self.mdp.state_actions
        for state in state_actions.keys():
            for action in state_actions[state]:
                self.q[(state[0],state[1],action)] = 0

    def run_episode(self):
        last_action = ''
        while not self.mdp.is_terminal(last_action):
            state1 = self.mdp.current_state
            action = self.choose_action(state1)
            action1 = action[0]
            last_action = action1
            res_do_action = self.mdp.do_action(action[0], self.mdp.current_state)
            state2 = res_do_action[1]
            action2 = self.choose_best_action(state2)[0]
            reward = res_do_action[0]
            self.action_function(state1,action1,reward,state2,action2)
            self.mdp.current_state = state2
    
    def run_value_iteration(self):
        # Begins at iteration 2 because first iteration is initializing rewards
        converge = 0
        i = 1
        while self.iterations >= i:
            i += 1
            self.mdp.init_state()
            self.run_episode()
        print("Total iterations: " + str(i))
    
    def action_function(self,state1,action1,reward,state2,action2):
        self.q[(state1[0], state1[1], action1)] = (1-self.alpha)*self.q[(state1[0], state1[1], action1)] + self.alpha*(reward + self.discount*self.q[(state2[0], state2[1], action2)])
    
    def choose_best_action(self, state):
        possible_actions = self.mdp.get_possible_actions(state)
        best_actions = []
        best_q_value = -9999999
        for action in possible_actions:
            if len(best_actions) == 0:
                best_actions.append(action)
                best_q_value = self.q[(state[0], state[1], action)]
            else:
                if best_q_value == self.q[(state[0], state[1], action)]:
                    best_actions.append(action)
                elif best_q_value < self.q[(state[0], state[1], action)]:
                    best_actions = [action]
                    best_q_value = self.q[(state[0], state[1], action)]
                    
        best_action = random.choice(best_actions)
        return [best_action, best_action]
    
    def choose_action(self, state):
        possible_actions = self.mdp.get_possible_actions(state).copy()
        best_actions = []
        best_q_value = -9999999
        for action in possible_actions:
            if len(best_actions) == 0:
                best_actions.append(action)
                best_q_value = self.q[(state[0], state[1], action)]
            else:
                if best_q_value == self.q[(state[0], state[1], action)]:
                    best_actions.append(action)
                elif best_q_value < self.q[(state[0], state[1], action)]:
                    best_actions = [action]
                    best_q_value = self.q[(state[0], state[1], action)]
                    
        best_action = random.choice(best_actions)
        if random.random() < (1-self.epsilon) :
            return [best_action, best_action]
        else:
            possible_actions.remove(best_action)
            return [random.choice(possible_actions), best_action]

grid = Gridworld(cells={(0, 0): 'lt', (0, 1): 't', (0, 3): 't', (0, 4): 'tr', (0, 5): 'lt', (0, 6): 't', (0, 7): 't', (0, 8): 't', (0, 9): 'tr', (1, 0): 'l', (1, 4): 'r', (1, 5): 'l', (1, 9): 'r', (2, 0): 'l', (2, 9): 'r', (3, 0): 'l', (3, 4): 'r', (3, 5): 'l', (3, 9): 'r', (4, 0): 'lb', (4, 4): 'rb', (4, 5): 'lb', (4, 9): 'rb', (5, 0): 'lt', (5, 1): 't', (5, 3): 't', (5, 4): 'tr', (5, 5): 'tl', (5, 6): 't', (5, 8): 't', (5, 9): 'tr', (6, 0): 'l', (6, 4):'r' , (6, 5): 'l', (6, 9): 'r', (7, 0): 'l', (7, 9): 'r', (8, 0): 'l',(8, 4): 'r', (8, 5): 'l', (8, 9): 'r', (9, 0): 'lb', (9, 1): 'b', (9, 2): 'b', (9, 3): 'b', (9, 4): 'rb', (9, 5): 'lb', (9, 6): 'b', (9, 7): 'b', (9, 8): 'b', (9, 9): 'rb', (0, 2): 'te'},
                 dimensions = [10,10])

iteration = QLearning(grid, discount = 0.7, alpha=0.1, iterations = 10000, epsilon=0.1)
iteration.run_value_iteration()

board = np.full((10, 10), float('-inf'))
for (x, y, action), value in iteration.q.items():
    board[x][y] = max(board[x][y], value)
show_values(board)