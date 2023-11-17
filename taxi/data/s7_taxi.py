# Luccas Rojas 201923052

import random
import pandas as pd

class QLearning():
    def __init__(self, board):
        self.current_state = (0,0)
        self.height = len(board.grid)
        self.width = len(board.grid[0])
        self.depth = len(board.grid[0][0])
        self.env = board
        self.discount = 0.60
        self.learning_rate = 0.2
        self.epsilon = 0.7
        self.policy = [[[random.choice(self.env.get_posible_actions((i,j,k)))for k in range(self.depth)] for j in range(self.width)] for i in range(self.height)]
        self.Q = [[[{"up":0,"down":0,"right":0,"pick":0,"drop":0} if ((i,j,k) in board.left_border) else {"up":0,"down":0,"left":0,"pick":0,"drop":0} if ((i,j,k) in board.right_border) else {"up":0,"down":0,"left":0,"right":0,"pick":0,"drop":0} for k in range(self.depth)] for j in range(self.width) ] for i in range(self.height)]
        
    def print_policy(self, policy):
        # Without passenger
        print("Sin pasajero")
        matrix = []
        for i in range(len(policy)):
            fila = []
            for j in range(len(policy[i])): 
                action = policy[i][j][0]
                fila.append(action)
            matrix.append(fila)
        data_frame = pd.DataFrame(matrix)
        print('\n')
        print(data_frame)
        print('\n')
        # With passenger
        print("Con pasajero")
        matrix = []
        for i in range(len(policy)):
            fila = []
            for j in range(len(policy[i])): 
                action = policy[i][j][1]
                fila.append(action)
            matrix.append(fila)
        data_frame = pd.DataFrame(matrix)
        print(data_frame)
        print('\n')
    
    def run(self):
        counter =0
        centinela = True
        while centinela:
            new_policy = [[[self.policy[i][j][k]for k in range(self.depth)] for j in range(self.width)] for i in range(self.height)]
            state = (random.randint(0,4),random.randint(0,4),0)
            board.set_current_state(state)
            while state != (-1,-1,-1):
                action1 = self.get_best_possible_action(state)
                action1 = self.choose_action(action1,state)
                reward,state2 = board.do_action(state,action1)
                if state2 == (-1,-1,-1):
                    action2 = "exit"
                else:
                    action2 = self.get_best_possible_action(state2)
                q_value =self.q_value_function(state,action1,reward,state2,action2)
                self.Q[state[0]][state[1]][state[2]][action1] = q_value

                state = state2

            centinela = counter<280
            new_policy = self.calculate_policy()
            # centinela =  not self.converge(self.policy,new_policy) 
            self.policy = new_policy

            counter+=1
            print("Episode: ",counter)
            # self.print_policy(self.policy)

    def calculate_policy(self):
        new_policy = [[[self.policy[i][j][k] for k in range(self.depth)] for j in range(self.width)] for i in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.depth):
                    action = self.get_best_possible_action((i,j,k))
                    new_policy[i][j][k] = action
        return new_policy
    
    def get_best_possible_action(self,state):
        best_action = self.policy[state[0]][state[1]][state[2]]
        best_actions = []
        for action in self.env.get_posible_actions(state):
            if self.Q[state[0]][state[1]][state[2]][action] > self.Q[state[0]][state[1]][state[2]][best_action]:
                best_actions = []
                best_actions.append(action)
            elif self.Q[state[0]][state[1]][state[2]][action] == self.Q[state[0]][state[1]][state[2]][best_action]:
                best_actions.append(action)
        if len(best_actions) > 0:
            best_action = random.choice(best_actions)
        return best_action
    
    def choose_action(self,action,state):
        if random.random() < self.epsilon:
            if action != "exit":
                possible_actions = self.env.get_posible_actions(state)
                possible_actions.remove(action)
                return random.choice((possible_actions))
            else:
                return action
        else:
            return action
        
    def q_value_function(self,state1,action1,reward,state2,action2):
        actual_q_value = self.Q[state1[0]][state1[1]][state1[2]][action1]
        if state2 == (-1,-1,-1):
            best_next_q_value = 0
        else:
            best_next_q_value = self.Q[state2[0]][state2[1]][state2[2]][action2]
        return (1-self.learning_rate)*actual_q_value + self.learning_rate*(reward + self.discount*best_next_q_value)
    
    def converge(self,policy,new_policiy):
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.depth):
                    if policy[i][j][k]!= new_policiy[i][j][k]:
                        return False


        return True
        
class Board():
    def __init__(self,grid,left_border,right_border,pick,drop):
        self.current_state=(0,0,0)
        self.height = len(grid)
        self.width = len(grid[0])
        self.depth = len(grid[0][0])
        self.grid = grid
        self.left_border = left_border
        self.right_border = right_border
        self.pick = pick
        self.drop = drop
        
    def set_current_state(self,state):
        self.current_state = state

    #Get the current state of the agent in the board   
    def get_current_state(self):
        return self.current_state

    #Get the posible actions 
    def get_posible_actions(self,current_state):
        posible_actions = []
        if current_state == (-1,-1,-1):
            return ["",""]
        elif (current_state[0],current_state[1]) in  self.left_border:
            posible_actions = ["up","down","right","pick","drop"]
        elif (current_state[0],current_state[1]) in self.right_border:
            posible_actions = ["up","down","left","pick","drop"]
        else:
            posible_actions = ["up","down","left","right","pick","drop"]
        return posible_actions

    #Goes to the specified state and returns the reward of going to that state    
    def do_action(self,state,action):
        reward = 0
        self.current_state = state
        noise_factor = random.uniform(0,1)

        if action == "up":
            if state[0]>0:
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
                self.current_state = (self.current_state[0]-1,self.current_state[1],self.current_state[2])
            else:
                self.current_state = (state)
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
        elif action == "down":
            if state[0]<self.height-1:
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
                self.current_state = (self.current_state[0]+1,self.current_state[1],self.current_state[2])
            else:
                self.current_state = (state)
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
        elif action == "left":
            if state[1]>0:
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
                self.current_state = (self.current_state[0],self.current_state[1]-1,self.current_state[2])
            else:
                self.current_state = (state)
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
        elif action == "right":
            if state[1]<self.width-1:
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]
                self.current_state = (self.current_state[0],self.current_state[1]+1,self.current_state[2])
            else:
                self.current_state = (state)
                reward = self.grid[self.current_state[0]][self.current_state[1]][self.current_state[2]]

        elif action == "pick":
            if state == self.pick:
                reward = 1
                self.current_state = (state[0],state[1],1)
            else:
                reward = -10
                self.current_state = (state[0],state[1],state[2])
        elif action == "drop":
            if state == self.drop:
                reward = 5
                self.current_state = (-1,-1,-1)
            else:
                reward = -10
                self.current_state = (state[0],state[1],state[2])

        return (reward,self.current_state)

#Creacion del laberinto de cuartos
height = 5
width = 5
# 0 quiere decir que no tiene pasajero y 1 que si
depth = 2
gridworld = [[[-0.1 for _ in range(depth)] for _ in range(width)] for _ in range(height)]
pick = (4,3,0)
drop = (4,0,1)
left_border = [(0,2),(1,2),(3,1),(4,1),(3,3),(4,3)]
right_border = [(0,1),(1,1),(4,2),(3,2),(3,0),(4,0)]

board = Board(gridworld,left_border,right_border,pick,drop)

q_learning = QLearning(board)
q_learning.run()
q_learning.print_policy(q_learning.policy)
print(q_learning.Q)