import numpy as np
import random
from tabulate import tabulate

class Taxi_Env:    
  def __init__(self, dimensions:tuple=(5,8), blocked_states=[(2, 2)], stations={}):
    self.dimensions = dimensions
    self.height = self.dimensions[0]
    self.width = self.dimensions[1] 
    self.board = [[' ' for _ in range(self.width)] for _ in range(self.height)]
    self.blocked_states = blocked_states
    self.stations = stations
    self.states = [(x,y) for x in range(self.width) for y in range(self.height)]

    #definir aleatoriamente inicio y destino del pasajero
    keys = list(self.stations.keys())
    random.shuffle(keys)
    self.start_state = keys[0]
    self.goal_state = self.start_state
    while self.goal_state == self.start_state :
        self.goal_state = keys[1]

    #setear estaciones
    for key in self.stations.keys():
      i,j = key
      self.board[i][j] = self.stations[(i,j)]
    #setear celdas bloqueadas
    for i,j in blocked_states:
        self.set_obstacle(i, j)

  def set_obstacle(self, row, column):
      self.board[row][column] = '*'
                                                                
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

class Q_Taxi_Agent():
    def __init__(self, environment):
      self.environment = environment
      self.status = [[0 for _ in range(self.environment.width)] for _ in range(self.environment.height)]
      self.actions = {'up', 'down', 'left', 'right', 'pickup', 'dropoff'}
      self.passenger_pickup = False 
      self.passenger_dropoff = False
      self.pickup = 0

      #inicio aleatorio del agente-taxi
      self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))
      while self.current_state in self.environment.blocked_states:
        self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))

    def get_current_state(self):
        return self.current_state

    def get_possible_actions(self, state):
        i, j = state
        actions = []
        if state in self.environment.stations:
            actions.append('dropoff')
            actions.append('pickup')
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
        elif action == 'pickup' and self.current_state==self.environment.start_state:
            self.passenger_pickup = True
        elif action == 'dropoff' and self.current_state==self.environment.goal_state:
            self.passenger_dropoff = True
        self.current_state = (i, j)
        return self.current_state
        
    def reset(self):
        self.passenger_pickup = False 
        self.passenger_dropoff = False 
        self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))
        while self.current_state in self.environment.blocked_states:
          self.current_state = (random.randint(0, self.environment.height-1), random.randint(0, self.environment.width -1))
        print('new start state:', self.current_state)


class Q_Taxi_Learning:
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
            self.q_table[(x,y)] = {'up':0, 'down':0, 'left':0, 'right':0, 'pickup':0, 'dropoff':0}
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
    old_passanger_state_dropoff = self.agent.passenger_dropoff
    next_state = self.agent.do_action(action)
    reward, done = self.getRewards(action, next_state)
    return next_state, reward, done
  
  def getRewards(self, action, state):
    #reward = - 1 #-1 por moverse
    reward = 0
    
    if action == 'pickup': 
      if state==self.environment.start_state:
        reward += 1
      else:
        reward += -10

    if action == 'dropoff': 
      if self.agent.passenger_dropoff == True and state == self.environment.goal_state:
        reward += 5
      else:
        reward += -10

    if state == self.environment.goal_state and self.agent.passenger_pickup == True and self.agent.passenger_dropoff == True: 
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
          else:
            cell = str(policy[i, j])
          width = len(cell)
          row.append('{:^{}}'.format(cell, width))
        table.append(row)
    print(tabulate(table, tablefmt='fancy_grid'))

#definir el ambiente taxi
taxi_env = Taxi_Env(dimensions=(5, 7), 
                blocked_states=[(0,2), (1,2), (3,1), (4,1), (3,4), (4,4)],
                stations={(0, 0): 'R', (0, 6): 'G', (4, 0): 'Y', (4, 6): 'B'})

taxi_env.print_board()
print("ubicación de pasajero", taxi_env.start_state, "=", taxi_env.stations[taxi_env.start_state])
print("destino de pasajero", taxi_env.goal_state, "=", taxi_env.stations[taxi_env.goal_state])

taxi_agent = Q_Taxi_Agent(taxi_env)
taxi_learn = Q_Taxi_Learning(taxi_env, taxi_agent, epsilon=0.5, alpha=0.1, gamma=0.9)

episodes = 5
for i in range(0, episodes):
    print(f"Episode: {i+1}")
    taxi_learn.run()
    taxi_agent.reset()