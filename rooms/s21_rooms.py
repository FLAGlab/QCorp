class Laberinto:
    def __init__(self):
        self.board = [[0]*10 for i in range(10)]
        self.board[0][2] = True
    
    def start(self):
        return (8, 6)
    
    def end(self): 
        return (0, 2)
    

import numpy as np
import random
class Learner:
    def __init__(self, agent, env, alpha=0.1, gamma=0.6, epsilon=0.1):
        #hyper parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.environment = env
        self.agent = agent          #actual agent
        self.qtable = self.__initdic__() #rewards table
    
    def __initdic__(self):
        table = dict()
        n = len(self.environment.board)
        for i in range(n):
            for j in range(n):
                table[(i,j)] = np.zeros(len(self.agent.actions))
        return table

    def run(self):
        done = False
        while not done:
            current_state = self.agent.state
            if random.uniform(0,1) < self.epsilon:
                action = self.randomAction()
            else:
                action = np.argmax(self.qtable[current_state]) 
            next_state, reward, done, info = self.step(action)
            old_value = self.qtable[current_state][action]
            next_max = np.max(self.qtable[next_state])
            new_value = (1 - self.alpha)*old_value + self.alpha*(reward + self.gamma*next_max)
            self.qtable[current_state][action] = new_value

            #print(info)
            #print(f'{current_state}, {action}, {next_state}')

    def randomAction(self):
        # return random.randint(0,len(self.agent.actions)-1)
        return np.random.choice(self.agent.get_posible_actions())

    def step(self, action):
        old_state = self.agent.state
        reward, done = self.getRewards(old_state, self.agent.getAction(action))
        self.agent.action(action)
        next_state = self.agent.state
        info = f'Executed action: {self.agent.getAction(action)} at state {old_state}'
        return next_state, reward, done, info

    def getRewards(self, state, action):
        if state == self.environment.end() and action == 'up':
            return 10, True
        else:
            return 0, False
        


class Agent:
    def __init__(self, env):
        self.state = env.start()
        self.initial_state = env.start()
        self.actions = [0,1,2,3]  #0-izq, 1-der, 2-arriba, 3-abajo
        self.bound = len(env.board) - 1   
        
    def get_posible_actions(self):
        actions = []
        current_state = self.state
        for i in range(len(self.actions)):
            self.action(i)
            if current_state != self.state:
                actions.append(i)
                self.state = current_state
        return actions
        

    def forward(self): #dere
        if not (self.state[1] == 4 and self.state[0] not in (2, 7)):
            self.state = (self.state[0], min(self.state[1] + 1, self.bound))
        
    def back(self): #izquierada
        if not (self.state[1] == 5 and self.state[0] not in (2, 7)):
            self.state = (self.state[0], max(self.state[1] - 1, 0))
        
    def upward(self): # arriba
        if not (self.state[0] == 5 and self.state[1] not in (2, 7)):
            self.state = (min(self.state[0] + 1, self.bound), self.state[1])
        
    def downward(self): # Bajar
        if not (self.state[0] == 4 and self.state[1] not in (2, 7)):
            self.state = (max(self.state[0] - 1, 0), self.state[1])
        
    def action(self, action : int):
        if action == 1:
            self.forward()
        elif action == 0:
            self.back()
        elif action == 2:
            self.upward()
        else:
            self.downward()
            
    def reset(self):
        self.state = self.initial_state
    
    def getAction(self, action : int):
        if action == 1:
            return 'right'
        elif action == 0:
            return 'left'
        elif action == 2:
            return 'up'
        else:
            return 'down'


def main():
    episodes = 500
    e = Laberinto()
    a = Agent(e)
    l = Learner(a, e, epsilon=0.2)
    for i in range(0, episodes):
        #print(f"Episode: {i+1}")
        l.run()
        a.reset()
    print(l.qtable)
        
main()