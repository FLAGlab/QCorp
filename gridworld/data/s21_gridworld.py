class Gridworld:
    def __init__(self):
        self.board = [[0]*10 for i in range(10)]
        self.board[5][5] = True
    
    def start(self):
        return (0, 0)
    
    def trap(self):
        return[(4,5), (7,5), (7,6)]
    
    def end(self): 
        return (5, 5)


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

            print(info) # Descomentar para imprimir accion y estado
            print(f'{current_state}, {action}, {next_state}')

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
        i,j =state
        if state == self.environment.end(): # hay dos manera de llegar arriba (Up) e izquierda(back).
            return 1, True
        elif (i,j) in self.environment.trap():
            return -1, True
        else: 
            return 0, False


class Agent:
    def __init__(self, env):
        self.state = env.start()
        self.initial_state = env.start()
        self.actions = [0,1,2,3]  #0-izq, 1-der, 2-arriba, 3-abajo
        self.bound = len(env.board) - 1   
        self.blocked_states = [(2, i) for i in range(1, 5)]
        self.blocked_states += [(2, i) for i in range(6, 9)]
        self.blocked_states += [(i, 4) for i in range(2, 8)]
        
        
    def get_posible_actions(self):
        actions = []
        current_state = self.state
        for i in range(len(self.actions)):
            self.action(i)
            if current_state != self.state:
                actions.append(i)
                self.state = current_state
        #print(actions) #descomentar para imprimir acción
        return actions
        

    def forward(self): # derecha
        new_state = (self.state[0], min(self.state[1] + 1, self.bound))
        if new_state not in self.blocked_states:
            self.state = new_state
        
    def back(self): #izquierda
        new_state = (self.state[0], max(self.state[1] - 1, 0))
        if new_state not in self.blocked_states:
            self.state = new_state
        
    def upward(self): # arriba
        new_state = (max(self.state[0] - 1, 0), self.state[1])
        if new_state not in self.blocked_states:
            self.state = new_state
        
    def downward(self): # abajo
        new_state = (min(self.state[0] + 1, self.bound), self.state[1])
        if new_state not in self.blocked_states:
            self.state = new_state
        
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
    episodes = 2
    e = Gridworld()
    a = Agent(e)
    l = Learner(a, e, epsilon=0.5)
    for i in range(0, episodes):
        print(f"Episode: {i+1}") # descomentar si se quiere imprimir el episodio
        l.run()
        a.reset()
    print(l.qtable)
        
main()