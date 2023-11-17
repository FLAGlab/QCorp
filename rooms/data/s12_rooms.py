import numpy as np
import pandas as pd
import random as rnd

class Rooms:

    def __init__(self, dimensions = 13):
        self.dimensions = dimensions
        self.board = [[0 for i in range (0, dimensions)] for j in range(0, dimensions)]
        self.current_state = (0,0)

        ## en qué momento se llenan las casillas con *, 0, 1, -1
        for i in range(0, dimensions):
            self.board[i][0] = '*'
            self.board[i][12] = '*'
            self.board[12][i] = '*'
            
            if i != 3 and i != 9:
                self.board[6][i] = '*'
                self.board[i][6] = '*'
                self.board[0][i] = '*'
        
        self.board[0][9] = '*'
        self.board[0][3] = 1

    def get_current_state(self):
        return self.current_state
    
    def set_current_state(self, state):
        self.current_state = (state[0], state[1])

    def get_posible_actions(self, i, j):

        states = []
        if i > 1 and self.board[i-1][j] != '*':
            states.append('north')
        if i < self.dimensions -1  and self.board[i+1][j] != '*':
            states.append('south')
        if j > 1 and self.board[i][j-1] != '*':
            states.append('east')
        if j < self.dimensions -1 and self.board[i][j+1] != '*':
            states.append('west')
        
        return states

    def do_action(self, action):

        if action == 'north':
            self.current_state = (self.current_state[0]-1, self.current_state[1])
        elif action == 'south':
            self.current_state = (self.current_state[0]+1, self.current_state[1])
        elif action == 'east':
            self.current_state = (self.current_state[0], self.current_state[1]-1)
        elif action == 'west':
            self.current_state = (self.current_state[0], self.current_state[1]+1)

        return (self.board[self.current_state[0]][self.current_state[1]], (self.current_state[0], self.current_state[1]))

    def reset(self):
        self.current_state = (0,0)
    
    def is_terminal(self):
        terminal = False
        if self.board[self.current_state[0]][self.current_state[1]] == 1 or self.board[self.current_state[0]][self.current_state[1]] == - 1:
            terminal = True
        return terminal

class qLearning:

    def __init__(self):
        self.mdp = Rooms()
        print(self.mdp.board)
        self.alpha = 0.7 # Tasa de aprendizaje
        self.gamma = 0.9 # Tasa de descuento 
        self.epsilon = 0.05 # Tasa de exploración

        self.actions = ['north', 'south', 'east', 'west', 'final']
        
        self.final_states = ['(0,3)']
        table = np.zeros((self.mdp.dimensions ** 2, len(self.actions)))
        labels_array = [['('+str(i)+','+str(j)+')' for i in range (0, self.mdp.dimensions)] for j in range(0, self.mdp.dimensions)]

        self.labels = []
        for column in labels_array:
            for element in column:
                self.labels.append(element)

        self.q_table = pd.DataFrame(table, columns = self.actions, index = self.labels)
        self.rewards = pd.DataFrame(table, columns = self.actions, index = self.labels)

        self.fill_rewards(self.mdp.dimensions)

        #print(self.rewards.to_string())
        # Se deben limitar las acciones de las casillas en la matriz que no se puedan acceder
    
    def fill_rewards(self, dimensions):
        for i in range(0, dimensions):
            for j in range(0, dimensions):

                generic_state = '(' + str(i) + ',' + str(j) + ')'

                if self.mdp.board[i][j] == '*':
                    for action in self.actions:
                        self.rewards.at[generic_state, action] = -100
                elif self.mdp.board[i][j] == 1:
                    self.rewards.at[generic_state,'final'] = 100
                    self.rewards.at[generic_state,'north'] = -100
                    self.rewards.at[generic_state,'south'] = -100
                    self.rewards.at[generic_state,'east'] = -100
                    self.rewards.at[generic_state,'west'] = -100
                else:
                    self.rewards.at[generic_state, 'final'] = -100
                    self.rewards.at[generic_state,'north'] = -5
                    self.rewards.at[generic_state,'south'] = -5
                    self.rewards.at[generic_state,'east'] = -5
                    self.rewards.at[generic_state,'west'] = -5


                # north
                if i - 1 < 0 or self.mdp.board[i-1][j] == '*' :
                    self.rewards.at[generic_state, 'north'] = -100
                #elif self.mdp.board[i-1][j] == 1 or self.mdp.board[i-1][j] == -1:
                    #self.rewards.at[generic_state, 'north'] = self.mdp.board[i-1][j]
                #    self.rewards.at[generic_state, 'final'] = self.mdp.board[i-1][j]
                #else:
                #    self.rewards.at[generic_state, 'final'] = -100

                # south
                if i + 1  > 9 or self.mdp.board[i+1][j] == '*' :
                    self.rewards.at[generic_state, 'south'] = -100

                # east
                if j - 1 < 0 or self.mdp.board[i][j-1] == '*' :
                    self.rewards.at[generic_state, 'east']= -100

                # west
                if j + 1  > 9 or self.mdp.board[i][j+1] == '*' :
                    self.rewards.at[generic_state, 'west'] = -100            

    def train(self, episodes):
        # se elige un estado aleatorio.

        iterator = 0
        current_state = [0,0]
        max_state = [0,0]

        while iterator < episodes:

            valid_state = False
            while not valid_state:
                current_state[0] = np.random.randint(0, self.mdp.dimensions)
                current_state[1] = np.random.randint(0, self.mdp.dimensions)
                if self.mdp.board[current_state[0]][current_state[1]] != '*' and self.mdp.board[current_state[0]][current_state[1]] != 1:
                    valid_state = True

            string_current_state = '(' + str(current_state[0]) + ',' + str(current_state[1]) + ')'
            string_max_state = ''
            _max = 0

            while True:

                valid_random_state = False
                while not valid_random_state:
                    random_action = np.random.choice(['north', 'south', 'east', 'west'])
                    if self.rewards.at[string_current_state, random_action] != -100 :
                        valid_random_state = True
                    if (random_action == 'north' and current_state[0] == 0) or (random_action == 'south' and current_state[0] == 11) or (random_action == 'east' and current_state[1] == 0) or (random_action == 'west' and current_state[1] == 11):
                        valid_random_state = False

                if random_action == 'north':
                    string_max_state = '(' + str(current_state[0] - 1) + ',' + str(current_state[1]) + ')'
                    max_state[0] = current_state[0] - 1
                    max_state[1] = current_state[1]
                elif random_action == 'south':
                    string_max_state = '(' + str(current_state[0] + 1) + ',' + str(current_state[1]) + ')'
                    max_state[0] = current_state[0] + 1
                    max_state[1] = current_state[1]
                    _max = 1
                elif random_action == 'east':
                    string_max_state = '(' + str(current_state[0]) + ',' + str(current_state[1] - 1) + ')'
                    max_state[0] = current_state[0]
                    max_state[1] = current_state[1] - 1
                    _max = 2
                elif random_action == 'west':
                    string_max_state = '(' + str(current_state[0]) + ',' + str(current_state[1] + 1) + ')'
                    max_state[0] = current_state[0]
                    max_state[1] = current_state[1] + 1
                    _max = 3
                
                #print(self.q_table.loc[string_current_state])
                action_max = self.q_table.loc[string_current_state].idxmax()

                if rnd.uniform(0,1) < self.epsilon:
                    action_max = np.random.choice(['north', 'south', 'east', 'west'])
                #print(action_max)

                #print('reward: ', self.mdp.board[max_state[0]][max_state[1]])
                
                self.q_table.at[string_current_state, random_action] = (1 - self.alpha) * self.q_table.at[string_current_state, random_action] + self.alpha * (self.mdp.board[max_state[0]][max_state[1]] + self.gamma * self.q_table.at[string_current_state, action_max])
                if string_max_state in self.final_states:
                    break

                string_current_state = string_max_state
                current_state = max_state
            iterator = iterator + 1
        
        print(self.q_table.to_string())

    def get_path(self):
        
        table = np.zeros((self.mdp.dimensions ** 2, 1))
        paths = pd.DataFrame(table, index = self.labels, columns= ['path'])

        for state in self.labels:
            action_max = self.q_table.idxmax(axis=1)
            path = state

            while True:
                print('state ',state)
                print(self.q_table.loc[state].idxmax())
                action = action_max[state]

                if self.rewards.at[state, action] != -100:

                    print('action ', action)  
                    if action == 'north':
                        if len(state) == 5:
                            next_state = '(' + str(int(state[1])-1) + ',' + str(state[3]) + ')' 
                        elif len(state) == 6 and state[3] == ',':
                            next_state = '(' + str(int(state[1:2])-1) + ',' + str(state[4]) + ')'
                        elif len(state) == 6 and state[2] == ',':
                            next_state = '(' + str(int(state[1])-1) + ',' + str(state[3:4]) + ')'
                        else:
                            next_state = '(' + str(int(state[1:2])-1) + ',' + str(state[4:5]) + ')'

                        if next_state in path:
                            break 
                    elif action == 'south':
                        if len(state) == 5:
                            next_state = '(' + str(int(state[1])+1) + ',' + str(state[3]) + ')' 
                        elif len(state) == 6 and state[3] == ',':
                            next_state = '(' + str(int(state[1:2])+1) + ',' + str(state[4]) + ')'
                        elif len(state) == 6 and state[2] == ',':
                            next_state = '(' + str(int(state[1])+1) + ',' + str(state[3:4]) + ')'
                        else:
                            next_state = '(' + str(int(state[1:2])+1) + ',' + str(state[4:5]) + ')'

                        if next_state in path:
                            break 
                    elif action == 'east':
                        if len(state) == 5:
                            next_state = '(' + str(int(state[1])) + ',' + str(int(state[3])-1) + ')' 
                        elif len(state) == 6 and state[3] == ',':
                            next_state = '(' + str(int(state[1:2])) + ',' + str(int(state[4])-1) + ')'
                        elif len(state) == 6 and state[2] == ',':
                            next_state = '(' + str(int(state[1])) + ',' + str(int(state[3:4])-1) + ')'
                        else:
                            next_state = '(' + str(int(state[1:2])) + ',' + str(int(state[4:5])-1) + ')'

                        if next_state in path:
                            break      
                    elif action == 'west':
                        if len(state) == 5:
                            next_state = '(' + str(int(state[1])) + ',' + str(int(state[3])+1) + ')' 
                        elif len(state) == 6 and state[3] == ',':
                            next_state = '(' + str(int(state[1:2])) + ',' + str(int(state[4])+1) + ')'
                        elif len(state) == 6 and state[2] == ',':
                            next_state = '(' + str(int(state[1])) + ',' + str(int(state[3:4])+1) + ')'
                        else:
                            next_state = '(' + str(int(state[1:2])) + ',' + str(int(state[4:5])+1) + ')'

                        if next_state in path:
                            break   
                    else:
                        path += ']'
                        print(path)
                        break

                    path += ',' + next_state
                    state = next_state
                else: break

                print("end cycle")
            paths.at[state, 'path'] = path
        
        print(paths.to_string())

ql = qLearning()
ql.train(1000)
ql.get_path()

