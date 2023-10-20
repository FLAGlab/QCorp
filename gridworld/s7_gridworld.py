# Luccas Rojas 201923052

import random
import pandas as pd

class QLearning():
    def __init__(self, board, exits):
        self.current_state = (0,0)
        self.height = len(board.grid)
        self.width = len(board.grid[0])
        self.env = board
        self.exits = exits
        self.discount = 0.90
        self.learning_rate = 0.2
        self.epsilon = 0.7
        self.policy = [[random.choice(self.env.get_posible_actions((i,j))) for j in range(self.width)] for i in range(self.height)]
        self.Q = [[ {"up":0,"down":0,"left":0,"right":0} if ((i,j) not in board.exits and ((i,j) not in board.blanks)) else {"exit":0} if ((i,j) not in board.blanks) else {} for j in range(self.width) ] for i in range(self.height)]
        self.start_states = [(i,j) for i in range(self.width)  for j in range(self.height) if (i,j) not in self.env.blanks]

    def print_policy(self, policy):
        matrix = []
        for i in range(len(policy)):
            fila = []
            for j in range(len(policy[i])): 
                action = policy[i][j]
                fila.append(action)
            matrix.append(fila)
        data_frame = pd.DataFrame(matrix)
        print('\n')
        print(data_frame)
    
    
    def run(self):
        counter =0
        centinela = True
        while centinela:
            new_policy = [[self.policy[i][j] for j in range(self.width)] for i in range(self.height)]
            state = random.choice(self.start_states)
            board.set_current_state(state)
            while state != (-1,-1):
                action1 = self.get_best_possible_action(state)
                action1 = self.choose_action(action1,state)
                reward,state2 = board.do_action(state,action1)
                if state2 == (-1,-1):
                    action2 = "exit"
                else:
                    action2 = self.get_best_possible_action(state2)
                q_value =self.q_value_function(state,action1,reward,state2,action2)
                self.Q[state[0]][state[1]][action1] = q_value

                best_action = new_policy[state[0]][state[1]]
                best_actions = []
                for action in self.env.get_posible_actions(state):
                    if self.Q[state[0]][state[1]][action] > self.Q[state[0]][state[1]][best_action]:
                        best_action = action
                        best_actions = []
                        best_actions.append(action)
                    elif self.Q[state[0]][state[1]][action] == self.Q[state[0]][state[1]][best_action]:
                        best_actions.append(action)
                new_policy[state[0]][state[1]] = random.choice(best_actions)
                state = state2

            # centinela = counter<1000
            new_policy = self.calculate_policy()
            centinela =  not self.converge(self.policy,new_policy) 
            self.policy = new_policy
            counter+=1
            print("Episode: ",counter)

    def calculate_policy(self):
        new_policy = [[self.policy[i][j] for j in range(self.width)] for i in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                if (i,j) not in self.env.blanks:
                    action = self.get_best_possible_action((i,j))
                    new_policy[i][j] = action
        return new_policy
    
    def get_best_possible_action(self,state):
        best_action = self.policy[state[0]][state[1]]
        best_actions = []
        for action in self.env.get_posible_actions(state):
            if self.Q[state[0]][state[1]][action] > self.Q[state[0]][state[1]][best_action]:
                best_actions = []
                best_actions.append(action)
            elif self.Q[state[0]][state[1]][action] == self.Q[state[0]][state[1]][best_action]:
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
        actual_q_value = self.Q[state1[0]][state1[1]][action1]
        if state2 == (-1,-1):
            best_next_q_value = 0
        else:
            best_next_q_value = self.Q[state2[0]][state2[1]][action2]
        return (1-self.learning_rate)*actual_q_value + self.learning_rate*(reward + self.discount*best_next_q_value)
    
    def converge(self,policy,new_policiy):
        for i in range(self.height):
            for j in range(self.width):
                if policy[i][j] != new_policiy[i][j]:
                    return False
        return True
        
class Board():
    noise = 0.2 
    def __init__(self,grid,exits,blanks,negatives,positives):
        self.current_state=(0,0)
        self.height = len(grid)
        self.width = len(grid[0])
        self.grid = grid
        self.exits = exits
        self.blanks = blanks
        self.negatives = negatives
        self.positives = positives
        for blank in blanks:
            grid[blank[0]][blank[1]]=("*",0)
        for positive in positives:
            grid[positive[0]][positive[1]]=(" ",1)
        for negative in negatives:
            grid[negative[0]][negative[1]]=(" ",-1)
        
    def set_current_state(self,state):
        self.current_state = state

    def print_grid(self):
        for row in self.grid:
            print(row)

    #Get the current state of the agent in the board   
    def get_current_state(self):
        return self.current_state

    #Get the posible actions 
    def get_posible_actions(self,current_state):
        posible_actions = []
        if current_state == (-1,-1) or self.grid[current_state[0]][current_state[1]][0] == "*":
            return ["",""]
        if current_state in self.exits:
            posible_actions = ["exit"]
        else:
            posible_actions = ["up","down","left","right"]
        return posible_actions

    #Goes to the specified state and returns the reward of going to that state    
    def do_action(self,state,action):
        reward = 0
        self.current_state = state
        noise_factor = random.uniform(0,1)
        if action == "up":
            if noise_factor < self.noise:
                random_move = random.choice(["left","right","down"])
                if random_move == "left":
                    if state[1]>0:
                        if self.grid[state[0]][state[1]-1][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0],self.current_state[1]-1)
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                else:
                    if state[1]<self.width-1:
                        if self.grid[state[0]][state[1]+1][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0],self.current_state[1]+1)
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
            else:
                if state[0]>0:
                    if self.grid[state[0]-1][state[1]][0] == "*":
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                    else:    
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        self.current_state = (self.current_state[0]-1,self.current_state[1])
                else:
                    self.current_state = (state)
                    reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                    
        elif action == "down":
            if noise_factor < self.noise:
                random_move = random.choice(["left","right","up"])
                if random_move == "left":
                    if state[1]>0:
                        if self.grid[state[0]][state[1]-1][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0],self.current_state[1]-1)
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                else:
                    if state[1]<self.width-1:
                        if self.grid[state[0]][state[1]+1][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0],self.current_state[1]+1)
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
            else:
                if state[0]<self.height-1:
                    if self.grid[state[0]+1][state[1]][0] == "*":
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                    else:    
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        self.current_state = (self.current_state[0]+1,self.current_state[1])
                else:
                    self.current_state = (state)
                    reward = self.grid[self.current_state[0]][self.current_state[1]][1]

        elif action == "left":
            if noise_factor < self.noise:
                random_move = random.choice(["up","down","right"])
                if random_move == "up":
                    if state[0]>0:
                        if self.grid[state[0]-1][state[1]][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0]-1,self.current_state[1])
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                else:
                    if state[0]<self.height-1:
                        if self.grid[state[0]+1][state[1]][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0]+1,self.current_state[1])
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
            else:
                if state[1]>0:
                    if self.grid[state[0]][state[1]-1][0] == "*":
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                    else:    
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        self.current_state = (self.current_state[0],self.current_state[1]-1)
                else:
                    self.current_state = (state)
                    reward = self.grid[self.current_state[0]][self.current_state[1]][1]

        elif action == "right":
            if noise_factor < self.noise:
                random_move = random.choice(["up","down","left"])
                if random_move == "up":
                    if state[0]>0:
                        if self.grid[state[0]-1][state[1]][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0]-1,self.current_state[1])
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                else:
                    if state[0]<self.height-1:
                        if self.grid[state[0]+1][state[1]][0] == "*":
                            self.current_state = (state)
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        else:    
                            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                            self.current_state = (self.current_state[0]+1,self.current_state[1])
                    else:
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
            else:
                if state[1]<self.width-1:
                    if self.grid[state[0]][state[1]+1][0] == "*":
                        self.current_state = (state)
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                    else:    
                        reward = self.grid[self.current_state[0]][self.current_state[1]][1]
                        self.current_state = (self.current_state[0],self.current_state[1]+1)
                else:
                    self.current_state = (state)
                    reward = self.grid[self.current_state[0]][self.current_state[1]][1] 

        elif action == "exit":
            reward = self.grid[self.current_state[0]][self.current_state[1]][1]
            self.current_state = (-1,-1)
        return (reward,self.current_state)
    

#Descomentar el caso necesario y cambiar el nombre de la variable para alternar entre Bridge y Gridwold

# Creacion del grid de gridworld
height = 10
width = 10
gridworld = [[(" ",0) for _ in range(width)] for _ in range(height)]   
blank_spots= [(2,1),(2,2),(2,3),(2,4),(2,6),(2,7),(2,8),(3,4),(4,4),(5,4),(6,4),(7,4)]
positive_reward_spots_gridworld =[(5,5)]
negative_reward_spots_gridworld =[(4,5),(7,5),(7,6)]
exits_gridworld = negative_reward_spots_gridworld + positive_reward_spots_gridworld
for blank in blank_spots:
    gridworld[blank[0]][blank[1]]=("*",0)
for positive in positive_reward_spots_gridworld:
    gridworld[positive[0]][positive[1]]=(" ",1)
for negative in negative_reward_spots_gridworld:
    gridworld[negative[0]][negative[1]]=(" ",-1)

board = Board(gridworld,exits_gridworld,blank_spots,negative_reward_spots_gridworld,positive_reward_spots_gridworld)
board.print_grid()


q_learning = QLearning(board,exits_gridworld)
q_learning.run()
q_learning.print_policy(q_learning.policy)
print(q_learning.Q)