class Taxi:
    def __init__(self):
        self.board = [[0]*5 for i in range(5)]
    
    def start(self):
        return (1, 3)
    
    def end(self): 
        return [(0, 0), (0, 4), (4, 0)]



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
                table[(i,j,0)] = np.zeros(len(self.agent.actions))
                table[(i,j,1)] = np.zeros(len(self.agent.actions))
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
        i, j, p = state
        if (i, j) in self.environment.end():
            if p == 1: # has passenger
                if action == 'drop':
                    return 5, True
                return -10, True
            return -10, True # Si el taxi llega sin pasajero a un estado final (-10 y termina)
        else:
            if p == 1: # has passenger
                if action == 'drop':
                    return -10, True
                return 0, False
            else:
                if action == 'peek':
                    return 1, False
                return 0, False
            


class Agent:
    def __init__(self, env):
        i, j = env.start()
        self.state = (i, j, 0) # parte de i,j sin pasajero
        self.initial_state = (i, j, 0)
        self.actions = [0,1,2,3,4,5]  #0-izq, 1-der, 2-arriba, 3-abajo, 4-dejar, 5-recoger 
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
        

    def forward(self):
        i, j, _ = self.state
        restrict = [(0,1), (1,1), (3, 0), (4, 0), (3, 2), (4, 2)]
        if (i, j) not in restrict:
            self.state = (self.state[0], min(self.state[1] + 1, self.bound), self.state[2])
            
        
    def back(self):
        i, j, _ = self.state
        restrict = [(0,2), (1,2), (3, 1), (4, 1), (3, 3), (4, 3)]
        if (i, j) not in restrict:
            self.state = (self.state[0], max(self.state[1] - 1, 0), self.state[2])
            
        
    def upward(self):
        self.state = (max(self.state[0] - 1, 0), self.state[1], self.state[2])
        
        
    def downward(self):
        self.state = (min(self.state[0] + 1, self.bound), self.state[1], self.state[2])
        
        
    def action(self, action : int):
        if action == 1:
            self.forward()
        elif action == 0:
            self.back()
        elif action == 2:
            self.upward()
        elif action == 3:
            self.downward()
        elif action == 4:
            self.drop()
        else:
            self.peek()
        
    # dejar pasajero
    def drop(self):
        if self.state[2] == 1: # tiene pasajero
            self.state = (self.state[0], self.state[1], 0)
    
    # recoger pasajero
    def peek(self):
        if self.state[2] == 0 and self.state[0] == 4 and self.state[1] == 3: # no tiene pasajero y está en B (4,3)
            self.state = (self.state[0], self.state[1], 1)
            
    def reset(self):
        self.state = self.initial_state
        
    
    def getAction(self, action : int):
        if action == 1:
            return 'right'
        elif action == 0:
            return 'left'
        elif action == 2:
            return 'up'
        elif action == 3:
            return 'down'
        elif action == 4:
            return 'drop'
        else:
            return 'peek'


def main():
    episodes = 400
    e = Taxi()
    a = Agent(e)
    l = Learner(a, e, epsilon=0.5)
    for i in range(0, episodes):
        #print(f"Episode: {i+1}")
        l.run()
        a.reset()
    print(l.qtable)
        
main()