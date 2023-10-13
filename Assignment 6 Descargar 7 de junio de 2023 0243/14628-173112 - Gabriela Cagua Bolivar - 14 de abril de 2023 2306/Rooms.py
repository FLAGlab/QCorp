import random

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

""" 
grid = Rooms()
end = False
times = 0

while not end:
    
    actions = grid.get_posible_actions(grid.get_current_state()[0],grid.get_current_state()[1])
    threshold = 1 / len(actions)
    rnd = random.random()

    chooseAction = False
    action = 1

    for i in range (1, len(actions)+1):
        if rnd < threshold * i and not chooseAction:
            action = i
            chooseAction = True

    print(grid.do_action(actions[action-1]))
    times = times + 1

    if grid.is_terminal(): 
        end = True

print(times)
"""

