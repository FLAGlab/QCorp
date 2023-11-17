import numpy as np
from collections import defaultdict
from tabulate import tabulate

class GridWorld:    
  def __init__(self, dimensions:tuple=(4,4), start_state=(0,0), goal_state=(3,3), blocked_states=[(2, 2)], reward_neg={}):
    self.dimensions = dimensions
    self.height = self.dimensions[0]
    self.width = self.dimensions[1] 
    self.board = [[0 for _ in range(self.width)] for _ in range(self.height)]

    self.start_state = start_state
    self.goal_state = goal_state
    self.blocked_states = blocked_states
    self.reward_neg = reward_neg
    self.states = [(x,y) for x in range(self.width) for y in range(self.height)]

    #Definir 1 para la celda de salida
    self.board[self.goal_state[0]][self.goal_state[1]] = 1
    #setear recompensas negativas
    for key,value in reward_neg.items():
        self.board[key[0]][key[1]] = value
    #setear celdas bloqueadas
    for i,j in blocked_states:
        self.set_obstacle(i, j)

  def set_obstacle(self, row, column):
      self.board[row][column] = '*'
        
  def set_reward(self, row, column, reward):
      self.board[row][column] = reward
                                                                
  def print_board(self):
    table = []
    for i in range(len(self.board)):
        row = []
        for j in range(len(self.board[i])):
            cell = str(self.board[i][j])
            width = len(cell)
            row.append('{:^{}}'.format(cell, width))
        table.append(row)
    print(tabulate(table, tablefmt='fancy_grid'))

  def reset(self):
    self.current_state = (0,0)

  def is_terminal(self):
    return self.goal_state == (self.current_state[0], self.current_state[1])  
  
class Q_Agent():
    def __init__(self, environment):
      self.environment = environment
      #self.start_state = self.environment.start_state
      #self.goal_state = self.environment.goal_state
      self.status = [[0 for _ in range(self.environment.width)] for _ in range(self.environment.height)]
      self.current_state = self.environment.start_state

      #self.actions = {'up': (-1,0), 'down': (1,0), 'left': (0,-1), 'right': (0,1)}
      self.actions = {'up', 'down', 'left', 'right'}

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        i, j = state
        actions = []
        if i > 0 and self.environment.board[i-1][j] != '*':
            actions.append('up')
        if i < self.environment.height-1 and self.environment.board[i+1][j] != '*':
            actions.append('down')
        if j > 0 and self.environment.board[i][j-1] != '*':
            actions.append('left')
        if j < self.environment.width -1 and self.environment.board[i][j+1] != '*':
          actions.append('right')
        return actions

    def do_action(self, action):
        i, j = self.current_state
        if action == 'up':
            i -= 1
        elif action == 'down':
            i += 1
        elif action == 'left':
            j -= 1
        elif action == 'right':
            j += 1
        self.current_state = (i, j)
        return self.current_state
        
    def reset(self):
        self.current_state = (0, 0)
    
    def random_reset(self):
        self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))
        while self.current_state in self.environment.blocked_states:
          self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))
        print('new start state:', self.current_state)

class Q_Learning:
  def __init__(self, environment, agent, epsilon=0.05, alpha=0.1, gamma=1):
    self.environment = environment
    self.agent = agent
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

    self.q_table = dict() 
    #incializar q_table en cero
    for x in range(self.environment.height): 
        for y in range(self.environment.width):
            self.q_table[(x,y)] = {'up':0, 'down':0, 'left':0, 'right':0}
    #asignar Nan cuando no es posible la accion
    for i in range(self.environment.height):
      self.q_table[(i, self.environment.width -1)]['right'] = np.nan
      self.q_table[(i, 0)]['left'] = np.nan
    for i in range(self.environment.width):
      self.q_table[(self.environment.height -1, i)]['down'] = np.nan
      self.q_table[(0, i)]['up'] = np.nan
    for block in self.environment.blocked_states:
      i, j = block
      if i > 0:
        self.q_table[(i-1, j)]['down'] = np.nan
      if i < self.environment.height - 1:
        self.q_table[(i+1, j)]['up'] = np.nan
      if j > 0:
        self.q_table[(i, j-1)]['right'] = np.nan
      if j < self.environment.width - 1:
        self.q_table[(i, j+1)]['left'] = np.nan
    #if (0, 0) in self.environment.blocked_states:
    #  self.q_table[(0, 0)] = {action: np.nan for action in self.agent.actions}

    
  def run(self):
    done = False
    while not done:
      current_state = self.agent.current_state
      available_actions = self.agent.get_possible_actions(current_state)
      action = self.choose_action(available_actions)
      next_state, reward, done = self.step(current_state, action)
      old_value = self.q_table[current_state][action]
      next_max = max(v for v in self.q_table[next_state].values() if not np.isnan(v))
      self.q_table[current_state][action] = (1 - self.alpha)*old_value + self.alpha*(reward + self.gamma*next_max)
      print(f'state: {current_state}, action: {action}, next_sate:{next_state}')

  def choose_action(self, available_actions):
      """Devuelve la acción óptima de la tabla Q-Value siguiendo E-greedy"""
      if np.random.uniform(0,1) < self.epsilon:
          action = available_actions[np.random.randint(0, len(available_actions))]
      else:
          q_values_of_state = self.q_table[self.agent.current_state]
          possible_actions = [k for k in q_values_of_state.keys() if k in available_actions]
          maxValue = max(filter(lambda x: x is not np.nan, [q_values_of_state[k] for k in possible_actions]))
          action = np.random.choice([k for k, v in q_values_of_state.items() if v == maxValue and k in possible_actions])
      return action
      
  def step(self, old_state, action):
    next_state = self.agent.do_action(action)
    reward, done = self.getRewards(next_state)
    return next_state, reward, done
  
  def getRewards(self, state):
    reward = self.environment.board[state[0]][state[1]] - 1 #El valor del grid restando 1 por hacer el movimiento
    if state == self.environment.goal_state:
      done = True
    else:
      done = False
    return reward, done

  def get_policy(self):
    policy = {}
    table = []
    for state, actions in self.q_table.items():
        if state in self.environment.blocked_states:
            continue
        best_action = max((v, k) for k, v in actions.items() if not np.isnan(v))[1]
        policy[state] = best_action
    for i in range(len(self.environment.board)):
        row = []
        for j in range(len(self.environment.board[i])):
          state = (i, j)
          if state in self.environment.blocked_states:
            cell = '*'
          elif state == self.environment.goal_state:
            cell = 'goal'
          else:
            cell = str(policy[i, j])
          width = len(cell)
          row.append('{:^{}}'.format(cell, width))
        table.append(row)
    print(tabulate(table, tablefmt='fancy_grid'))

env = GridWorld(dimensions=(10, 10), start_state=(0,0), goal_state=(5,5), 
                blocked_states=[(2, 1), (2, 2), (2, 3), (2, 4) ,(2, 6) ,(2, 7) ,(2, 8) ,(3, 4) ,(4, 4) ,(5, 4) ,(6, 4),(7, 4)], 
                reward_neg={(4, 5): -1, (7, 5): -1, (7, 6): -1})

env.print_board()
agent = Q_Agent(env)
l = Q_Learning(env, agent, epsilon=0.05, alpha=0.1, gamma=1)
episodes = 3000
for i in range(0, episodes):
    print(f"Episode: {i+1}")
    l.run()
    agent.reset()