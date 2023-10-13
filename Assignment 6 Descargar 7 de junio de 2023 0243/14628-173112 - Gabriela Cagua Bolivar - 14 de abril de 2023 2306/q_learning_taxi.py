from Taxi_grid import taxi_grid as grid
import numpy as np
import pandas as pd
import random as rnd

class qLearning:

    def __init__(self):
        self.mdp = grid()
        self.alpha = 0.4 # Tasa de aprendizaje
        self.gamma = 0.95 # Tasa de descuento 
        self.epsilon = 0.05 # Tasa de exploración

        directions = ['north', 'south', 'east', 'west']
        instructions = ['move', 'pick up', 'drop off']
        status = ['passenger', 'no passenger']

        self.actions = []

        for direction in directions:
            for instruction in instructions:
                action = direction + ' - ' + instruction
                self.actions.append(action)

        self.final_states = ['(0,0)', '(0,4)', '(4,0)', '(4,3)'] # revisar
        table = np.zeros(((self.mdp.dimensions ** 2) * 2, len(self.actions)))
        labels_array = [[['('+str(i)+','+str(j)+')'+ ' - ' + element for element in status] for i in range (0, self.mdp.dimensions)] for j in range(0, self.mdp.dimensions)]

        self.labels = []
        for column in labels_array:
            for element in column:
                for dir in element:
                    self.labels.append(dir)

        self.q_table = pd.DataFrame(table, columns = self.actions, index = self.labels)
        self.rewards = pd.DataFrame(table, columns = self.actions, index = self.labels)

        self.fill_rewards()
        # Se deben limitar las acciones de las casillas en la matriz que no se puedan acceder
    
    def fill_rewards(self):
        # los que tengan pick up y drop off y que no sean finales se llevan -10
        for label in self.labels:
            for column in self.actions:
                #print(column, str(column).__contains__('pick up'))
                #print(label[0:5], label[0:5] not in self.final_states)
                if (str(column).__contains__('pick up') or str(column).__contains__('drop off')) and label[0:5] not in self.final_states: 
                    self.rewards.at[label, column] = -10

        # hay que ponerle las restricciones de los bordes
        
        instructions = ['move', 'pick up', 'drop off']
        for label in self.labels:
            if label[1] == '0':
                for element in instructions:
                    self.rewards.at[label, 'north - ' + element] = 'x'
                    self.q_table.at[label, 'north - ' + element] = 'x'
            if label[1] == '4':
                for element in instructions:
                    self.rewards.at[label, 'south - ' + element] = 'x'
                    self.q_table.at[label, 'south - ' + element] = 'x'
            if label[3] == '0':
                for element in instructions:
                    self.rewards.at[label, 'east - ' + element] = 'x'
                    self.q_table.at[label, 'east - ' + element] = 'x'
            if label[3] == '4':
                for element in instructions:
                    self.rewards.at[label, 'west - ' + element] = 'x'
                    self.q_table.at[label, 'west - ' + element] = 'x'
        
        # restricciones de los obstaculos
        obstacles = ['(0,1)', '(1,1)', '(3,0)', '(4,0)', '(3,2)', '(4,2)']
        status = ['passenger', 'no passenger']
        instructions = ['move', 'pick up', 'drop off']

        for obstacle in obstacles:
            for stat in status:
                for instruction in instructions:
                    self.rewards.at[obstacle + ' - ' + stat, 'west - ' + instruction] = 'x'
                    self.q_table.at[obstacle + ' - ' + stat, 'west - ' + instruction] = 'x'

        obstacles = ['(0,2)', '(1,2)', '(3,1)', '(4,1)', '(3,3)', '(4,3)']

        for obstacle in obstacles:
            for stat in status:
                for instruction in instructions:
                    self.rewards.at[obstacle + ' - ' + stat, 'east - ' + instruction] = 'x'
                    self.q_table.at[obstacle + ' - ' + stat, 'east - ' + instruction] = 'x'
        
    def train(self, episodes):
        # se elige un estado aleatorio.

        iterator = 0

        states = self.final_states

        init_state = np.random.choice(states)
        self.compute_actions(init_state, 'init')
        states.remove(init_state)
        final_state = np.random.choice(states)
        self.compute_actions(final_state, 'end')

        self.fill_non_zero()
        print(self.rewards.to_string())
   
        while iterator < episodes:
            current_state = init_state
            visited_states = []

            while True:
                current_state_string = current_state + ' - ' + 'passenger'

                # choosing an arbitrary action
                valid_action = False
                while not valid_action:
                    random_action = np.random.choice(self.actions)
                    valid_action = self.validate_action(current_state, random_action)

                # where does this action take me to?
                x, y, new_state = self.compute_new_state(current_state, random_action)
                print(self.q_table.loc[current_state_string])

                action_max = self.get_max_index(self.q_table.loc[current_state_string]) 
                self.q_table.at[current_state_string, random_action] = (1 - self.alpha) * self.q_table.at[current_state_string, random_action] + self.alpha * (self.rewards.at[current_state_string, random_action] + self.gamma * self.q_table.at[current_state_string, action_max])
                print((1 - self.alpha) * self.q_table.at[current_state_string, random_action] + self.alpha * (self.rewards.at[current_state_string, random_action] + self.gamma * self.q_table.at[current_state_string, action_max]))
                     
                if new_state == final_state: # si ya se llego al estado final
                    print(self.q_table.to_string())
                    break

                current_state = new_state

            iterator = iterator + 1

        self.get_path(init_state= init_state)

    def fill_non_zero(self):
        for label in self.labels:
            for action in self.actions:
                if self.rewards.at[label, action] == 0:
                    self.rewards.at[label, action] = -0.5
    
    def get_max_index(self, array: list):
        
        max = -1000
        idmax = 0

        iter = 0
        for element in array:
            if element != 'x' and element > max:
                max = element
                idmax = iter
            
            iter = iter + 1
        
        return self.actions[idmax]


    def validate_action(self, state, action):
        if state[1] == '0' and 'north' in action:
            return False
        if state[1] == '4' and 'south' in action:
            return False
        if state[3] == '0' and 'east' in action:
            return False
        if state[3] == '4' and 'west' in action:
            return False
        if self.q_table.at[state + ' - passenger', action] == 'x':
            return False
        else: return True

    def compute_new_state(self, state, action):
        x = state[1]
        y = state[3]

        if 'north' in action:
            x = int(x) - 1
        elif 'south' in action:
            x = int(x) + 1
        elif 'east' in action:
            y = int(y) - 1
        elif 'west' in action:
            y = int(y) + 1
        
        return x, y, '(' + str(x) + ',' + str(y) + ')'


    def compute_actions(self, state, action):

        if action == 'init':
            if state == '(0,0)':
                self.rewards.at['(0,1) - no passenger', 'north - pick up'] = 1
                self.rewards.at['(1,0) - no passenger', 'east - pick up'] = 1
            elif state == '(4,0)':
                self.rewards.at['(3,0) - no passenger', 'south - pick up'] = 1
            elif state == '(0,4)':
                self.rewards.at['(0,3) - no passenger', 'west - pick up'] = 1
                self.rewards.at['(1,4) - no passenger', 'north - pick up'] = 1
            elif state == '(4,3)':
                self.rewards.at['(3,3) - no passenger', 'south - pick up'] = 1
                self.rewards.at['(4,4) - no passenger', 'east - pick up'] = 1
            
            for label in self.labels:
                for action in self.actions:
                    if 'pick up' in action and self.rewards.at[label, action] != 'x': # and state not in label 
                        self.rewards.at[label, action] = -10
                    if 'no passenger' in label and state not in label:
                        self.rewards.at[label, action] = -10


        # todos los otros pick up se deben bloquear y los no passenger
        elif action == 'end':
            if state == '(0,0)':
                self.rewards.at['(0,1) - passenger', 'north - drop off'] = 5
                self.rewards.at['(1,0) - passenger', 'east - drop off'] = 5
            elif state == '(4,0)':
                self.rewards.at['(3,0) - passenger', 'south - drop off'] = 5
            elif state == '(0,4)':
                self.rewards.at['(0,3) - passenger', 'west - drop off'] = 5
                self.rewards.at['(1,4) - passenger', 'north - drop off'] = 5
            elif state == '(4,3)':
                self.rewards.at['(3,3) - passenger', 'south - drop off'] = 5
                self.rewards.at['(4,4) - passenger', 'east - drop off'] = 5

            for label in self.labels:
                for action in self.actions:
                    if 'drop off' in action and state not in label and self.rewards.at[label, action] != 'x' and self.rewards.at[label, action] != 5:
                        self.rewards.at[label, action] = -10
                    #if 'no passenger' in label and state not in label:
                    #    self.rewards.at[label, action] = -10
            # todos los otros drop off se deben bloquear: no passenger se desbloquea cuando ya se llegó a la meta

    def get_path(self, init_state):

        path = init_state
        visited_states = [init_state]
        current_state = init_state

        while True: 
            action_max = self.get_max_index(self.q_table.loc[current_state + ' - passenger'])
            _, _, new_state = self.compute_new_state(current_state, action_max)

            while new_state in visited_states:
                self.q_table.at[current_state, action_max] = -20
                action_max = self.get_max_index(self.q_table.loc[current_state + ' - passenger'])
                _, _, new_state = self.compute_new_state(current_state, action_max)

            visited_states.append(new_state)

            path += ' -> ' + new_state
            print(path)
            current_state = new_state

            if new_state in self.final_states:
                break
        

ql = qLearning()
ql.train(50)
ql.get_path()
