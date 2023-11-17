import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt

class GridWorld:
  def __init__(self, length, width, obj_coo, trap_coo, block_coo, second_obj, start=[0,0]):
    self.length = length
    self.width = width
    self.obj_coo = obj_coo
    self.trap_coo = trap_coo
    self.current_state = start
    self.block_coo = block_coo
    self.second_obj = second_obj
    self.board = [[0 for _ in range(self.width)] for _ in range(self.length)]

    for i in range(len(self.board)):
      for j in range(len(self.board[i])):
        if [i,j] == self.obj_coo:
          self.board[i][j] = 1
        elif [i,j] == self.second_obj:
          self.board[i][j] = 1
        elif (i,j) in self.trap_coo:
          self.board[i][j] = -1
        elif (i,j) in self.block_coo:
          self.board[i][j] = '*'

  def get_current_state(self):
    return self.current_state
  
  def get_posible_actions(self):
    actions = ['up', 'down', 'right', 'left']
    if (self.current_state[0] - 1, self.current_state[1]) in self.block_coo or self.current_state[0]-1 < 0:
      actions.pop(actions.index('up'))
    if (self.current_state[0] + 1, self.current_state[1]) in self.block_coo or self.current_state[0]+1 > self.length-1:
      actions.pop(actions.index('down'))
    if (self.current_state[0], self.current_state[1] + 1) in self.block_coo or self.current_state[1]+1 > self.width-1:
      actions.pop(actions.index('right'))
    if (self.current_state[0], self.current_state[1] - 1) in self.block_coo or self.current_state[1]-1 < 0:
      actions.pop(actions.index('left'))
    
    return actions
  
  def do_action(self, action):
    if action not in self.get_posible_actions():
      raise Exception('No')
    if action == 'up' and self.current_state[0]-1 >= 0:
      self.current_state[0] -= 1
    elif action == 'right' and self.current_state[1]+1 <= self.width-1:
      self.current_state[1] += 1
    elif action == 'down' and self.current_state[0]+1 <= self.length-1:
      self.current_state[0] += 1
    elif action == 'left' and self.current_state[1]-1 >= 0:
      self.current_state[1] -= 1
    
    return (self.board[self.get_current_state()[0]][self.get_current_state()[1]], self.get_current_state())
  
  def reset(self):
    self.current_state = [0,0]
  
  def is_terminal(self):
    return True if self.get_current_state() == self.obj_coo else False
  

class Grid_lab(GridWorld):
  def __init__(self, length, width, obj_coo, trap_coo, block_coo, second_obj, start=[0, 0]):
    super().__init__(length, width, obj_coo, trap_coo, block_coo, second_obj, start)
  
  def get_posible_actions(self):
    actions = ['up', 'down', 'right', 'left']
    wall_up = [[5,0],[5,1],[5,3],[5,4],[5,5],[5,6],[5,8],[5,9]]
    wall_down = [[4,0],[4,1],[4,3],[4,4],[4,5],[4,6],[4,8],[4,9]]
    wall_right = [[0,4],[1,4],[3,4],[4,4],[5,4],[6,4],[8,4],[9,4]]
    wall_left = [[0,5],[1,5],[3,5],[4,5],[5,5],[6,5],[8,5],[9,5]]
    if (self.current_state[0] - 1, self.current_state[1]) in self.block_coo or self.current_state[0]-1 < 0 or self.current_state in wall_up:
      actions.pop(actions.index('up'))
    if (self.current_state[0] + 1, self.current_state[1]) in self.block_coo or self.current_state[0]+1 > self.length-1 or self.current_state in wall_down:
      actions.pop(actions.index('down'))
    if (self.current_state[0], self.current_state[1] + 1) in self.block_coo or self.current_state[1]+1 > self.width-1 or self.current_state in wall_right:
      actions.pop(actions.index('right'))
    if (self.current_state[0], self.current_state[1] - 1) in self.block_coo or self.current_state[1]-1 < 0 or self.current_state in wall_left:
      actions.pop(actions.index('left'))
    
    return actions
  
#   def reset(self, states):
#     states = np.array(states)
#     self.current_state = list(states[np.random.choice(len(states),1)][0])

class SARSA:
  def __init__(self, epsilon, grid):
    self.epsilon = epsilon
    self.grid = grid
    self.states = []
    for i in range(len(self.grid.board)):
      for j in range(len(self.grid.board[0])):
        if (i,j) not in self.grid.block_coo:
          self.states.append([i,j])

  def e_soft_action(self, a, epsilon, actions):
    if np.random.random() < epsilon:
      return np.random.choice(actions)
    else:
      return a

  def greedify(self, po, Q):
    for s in po.keys():
        big = -np.inf
        state = 0
        for key, value in Q[s].items():
          if value > big:
            big = value
            state = key
        a = state
        po[s] = a
    
    return po

  def run_episode(self, po, epsilon, discount, step_size, Q):
    a = self.e_soft_action(po[tuple(self.grid.current_state)], epsilon, self.grid.get_posible_actions())
    s = self.grid.current_state.copy()
    while True:
      next_reward, next_state = self.grid.do_action(a)
      # Agregamos a la recompensa -1 por cada paso para llegar rapido al objetivo
      #next_reward -= 1
      if self.grid.is_terminal():
        Q[tuple(next_state)][a] = 0
        a_prime = self.e_soft_action(po[tuple(next_state)], epsilon, self.grid.get_posible_actions())
        Q[tuple(s)][a] = Q[tuple(s)][a] + step_size*(next_reward + discount*Q[tuple(next_state)][a_prime] - Q[tuple(s)][a])
        po = self.greedify(po, Q)
        break
      else:
        a_prime = self.e_soft_action(po[tuple(next_state)], epsilon, self.grid.get_posible_actions())
        Q[tuple(s)][a] = Q[tuple(s)][a] + step_size*(next_reward + discount*Q[tuple(next_state)][a_prime] - Q[tuple(s)][a])
        po = self.greedify(po, Q)
        s = next_state.copy()
        a = a_prime

    return Q, po
  
  def SARSA_alg(self, step_size, discount):
    po = {}
    for s in self.states:
      self.grid.current_state = s
      po[tuple(s)] = np.random.choice(self.grid.get_posible_actions())
    Q = {}
    for s in self.states:
      self.grid.current_state = s
      Q[tuple(s)] = {}
      for a in self.grid.get_posible_actions():
        Q[tuple(s)][a] = 0
    iter = 0
    delta = float('inf')
    delta_vals = []
    while delta > 0.01:
      self.grid.reset()
      Q_prev = deepcopy(Q)
      Q, po = self.run_episode(po, self.epsilon, discount, step_size, Q)
      delta = 0
      for s in self.states:
        self.grid.current_state = s
        for a in self.grid.get_posible_actions():
          s_tuple = tuple(s)
          delta = max(delta, abs(Q[s_tuple][a] - Q_prev[s_tuple][a]))
      iter += 1
      print(iter)
      delta_vals.append(delta)
    
    V = {}
    for s in po.keys():
      big = -np.inf
      state = 0
      for key, value in Q[s].items():
        if value > big:
          big = value
          state = key
      V[s] = big
    
    for s in po.keys():
      state_copy = list(s)
      self.grid.current_state = state_copy
      V[tuple(self.grid.obj_coo)] = 100
      big_V = float('-inf')
      for action in self.grid.get_posible_actions():
        self.grid.current_state = state_copy[:] 
        _, next_state = self.grid.do_action(action)
        if V[tuple(next_state)] > big_V:
          big_V = V[tuple(next_state)]
          po[s] = action

    return V, po, delta_vals
  

class Q_learning(SARSA):
  def __init__(self, epsilon, grid):
    self.epsilon = epsilon
    self.grid = grid
    self.states = []
    for i in range(len(self.grid.board)):
      for j in range(len(self.grid.board[0])):
        if (i,j) not in self.grid.block_coo:
          self.states.append([i,j])

  def e_soft_action(self, a, epsilon, actions):
    if np.random.random() < epsilon:
      return np.random.choice(actions)
    else:
      return a
  
  def run_episode(self, Q, po, epsilon, discount, step_size):
    s = self.grid.current_state.copy()
    a = self.e_soft_action(po[tuple(s)], epsilon, self.grid.get_posible_actions())
    while True:
      next_reward, next_state = self.grid.do_action(a)
      if self.grid.is_terminal():
        Q[tuple(next_state)][a] = 0
        next_max = max(Q[tuple(next_state)].values())
        Q[tuple(s)][a] = Q[tuple(s)][a] + step_size*(next_reward + discount*next_max - Q[tuple(s)][a])
        po = self.greedify(po, Q)
        break
      else:
        next_max = max(Q[tuple(next_state)].values())
        Q[tuple(s)][a] = Q[tuple(s)][a] + step_size*(next_reward + discount*next_max - Q[tuple(s)][a])
        po = self.greedify(po, Q)
        s = next_state.copy()
        a = self.e_soft_action(po[tuple(next_state)], epsilon, self.grid.get_posible_actions())

    return Q, po
  
  def Q_learning_alg(self, step_size, discount):
    po = {}
    for s in self.states:
      self.grid.current_state = s
      po[tuple(s)] = np.random.choice(self.grid.get_posible_actions())
    Q = {}
    for s in self.states:
      self.grid.current_state = s
      Q[tuple(s)] = {}
      for a in self.grid.get_posible_actions():
        Q[tuple(s)][a] = 1
    iter = 0
    delta = float('inf')
    delta_vals = []
    while delta > 0.01:
      # Para correr el algoritmo del laberinto se debe descomentar el metodo de reset() con los estados como parametro y comentar el anterior
      self.grid.reset()
      #self.grid.reset(self.states)
      Q_prev = deepcopy(Q)
      Q, po = self.run_episode(Q, po, self.epsilon, discount, step_size)
      delta = 0
      for s in self.states:
        self.grid.current_state = s
        for a in self.grid.get_posible_actions():
          s_tuple = tuple(s)
          delta = max(delta, abs(Q[s_tuple][a] - Q_prev[s_tuple][a]))
      iter += 1
      print(iter)
      delta_vals.append(delta)
    
    V = {}
    for s in po.keys():
      big = -np.inf
      state = 0
      for key, value in Q[s].items():
        if value > big:
          big = value
          state = key
      V[s] = big
    
    for s in po.keys():
      state_copy = list(s)
      self.grid.current_state = state_copy
      V[tuple(self.grid.obj_coo)] = 100
      big_V = float('-inf')
      for action in self.grid.get_posible_actions():
        self.grid.current_state = state_copy[:] 
        _, next_state = self.grid.do_action(action)
        if V[tuple(next_state)] > big_V:
          big_V = V[tuple(next_state)]
          po[s] = action

    return Q, V, po, delta_vals

    
grid_l = Grid_lab(10, 10, [0,2], {}, {}, [])
agent_q = Q_learning(0.25, grid_l)
Q, V, po, delta_vals = agent_q.Q_learning_alg(0.2, 0.9)