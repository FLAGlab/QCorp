import copy
import numpy as np
import random

def get_area(current_state):
    actx=current_state[1]
    acty=current_state[0]
    left= actx-1
    right = actx+1
    down=acty+1
    up= acty-1
    return actx,acty,left,right,down,up

class Cell:
  def __init__(self,value,is_terminal,walls=[]):
    self.value=value
    self.is_terminal=is_terminal
    self.walls=walls
  
  def get_value(self):
    return self.value

  def set_value(self,value):
    self.value=value

import copy
import random
class TaxiEnvironment:
  def __init__(self, board, dimensions,actions,possible_stops):
    self.board=board
    self.dimensions=dimensions
    self.is_terminal=False
    self.actions=actions
    self.possible_stops=possible_stops
    self.reset()

  def get_dimensions(self):
    return self.dimensions
  
  def get_current_state(self):
    mapped = 0
    if self.current_state[2]=='r':
      mapped=1
    elif self.current_state[2]=='g':
      mapped=2
    elif self.current_state[2]=='b':
      mapped=3
    elif self.current_state[2]=='y':
      mapped=4
    resp = copy.deepcopy(self.current_state)
    resp[2]=mapped
    return resp

  def do_action(self,action):
    return self.do_action_final_action(action)

  def do_action_final_action(self,action):
    reward = self.reward_function(action)
    new_state = self.get_current_state()
    return [reward,new_state]

  def reset(self):
    self.is_terminal=False
    self.plan_trip()
    y=random.randrange(self.dimensions[0])
    x=random.randrange(self.dimensions[1])
    self.current_state=[y,x,'n']

  def plan_trip(self):
    # pos_stops = copy.deepcopy(self.possible_stops)
    # pass_location = random.randrange(len(pos_stops))
    # self.passenger_location=pos_stops.pop(pass_location)
    # destiny_location=random.randrange(len(pos_stops))
    # self.destiny=pos_stops.pop(destiny_location)
    self.passenger_location='b'
    self.destiny='r'

  def set_state(self,new_state):
    self.current_state=new_state

  def is_terminal(self):
    return self.board[self.current_state[0]][self.current_state[1]].is_terminal

  def is_state_terminal(self,current_state):
    return self.board[current_state[0]][current_state[1]].is_terminal

  def get_board(self):
    return self.board

  def is_valid(self):
    return self.board[self.current_state[0]][self.current_state[1]].get_value()!='*'
  
  def is_state_valid(self,current_state):
    return self.board[current_state[0]][current_state[1]].get_value()!='*'

  def reward_function(self, action):
    cur_stat = [self.current_state[0],self.current_state[1]]
    actx,acty,left,right,down,up = get_area(cur_stat)

    reward = 0
    # print('Celda actual:')
    # print(repr(self.board[acty][actx]))
    if action in self.board[acty][actx].walls:
      return 0
    if action=='left' and left>=0:
        self.current_state[1]-=1
    elif action=='right' and right<self.dimensions[1]:
        self.current_state[1]+=1
    elif action=='up' and up>=0:
        self.current_state[0]-=1
    elif action=='down' and down<self.dimensions[0]:
        self.current_state[0]+=1
    elif action=='pick':
      if self.board[acty][actx].get_value()==self.passenger_location:
        self.passenger_location=''
        self.current_state[2]=copy.deepcopy(self.destiny)
        reward = 1 
      else: 
        reward = -10
    elif action=='drop':
      if self.board[acty][actx].get_value()==self.current_state[2]:
        reward = 5
      else: 
        reward = -10
      self.is_terminal = True
    return reward
  

class Learner:
    def __init__(self, agent, env, qtable,prev_qtable, check_convergence, alpha=0.1, gamma=0.6, epsilon=0.1,episode_cap=100000):
        #hyper parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.environment = env
        self.agent = agent          #actual agent
        self.qtable = qtable #rewards table
        self.prev_qtable = prev_qtable
        self.check_convergence=check_convergence
        self.episode_cap = episode_cap
    
    def __initdic__(self):
        table = dict()
        for i in range(0, self.environment.end()):
            table[i] = np.zeros(len(self.agent.actions))
        return table

    def run_til_convergence(self):
      converged = False
      episodes=1
      while not converged:
        print(f'Episodio:{episodes}')
        self.run_episode()
        converged = self.check_convergence(self.qtable, self.prev_qtable, self.environment.dimensions) or episodes>self.episode_cap
        episodes+=1
        if episodes%(self.episode_cap/10)==0:
          self.epsilon=max(self.epsilon-0.1, 0.1)
      print('Q-tabla previa')
      print(self.prev_qtable)
      print('Q-tabla final')
      print(self.qtable)
      return episodes     

    def run_episode(self):
        done = False
        steps=1
        while not done:
            #print(f'steps:{steps}')
            current_state = copy.deepcopy(self.environment.get_current_state())
            if random.uniform(0,1) < self.epsilon:
                action = self.randomAction()
            else:
                tuple_state = tuple(current_state)
                acts = self.qtable[tuple_state]
                action = max(acts, key=acts.get)
            next_state, reward = self.step(action)
            current_tuple_state = tuple(current_state)
            next_tuple_state = tuple(next_state)

            old_value = self.qtable[current_tuple_state][action]
            self.prev_qtable[current_tuple_state][action]=old_value
            acts = self.qtable[next_tuple_state]
            next_max = acts[max(acts, key=acts.get)]
            new_value = old_value + self.alpha * (reward+(self.gamma*next_max)-old_value)
            self.qtable[current_tuple_state][action] = new_value
            done = self.environment.is_terminal
            steps+=1
        self.environment.reset()

    def randomAction(self):
        return random.choice(self.agent.actions)

    def step(self, action):
        reward, next_state = self.environment.do_action(action)
        # print(f'Executed action: {self.agent.getAction(action)} at state {old_state}')
        return next_state, reward  
    
class Agent:
    def __init__(self, actions):
        self.actions = actions

def check_policy_convergence(act,prev,dimensions):
  converged = True
  for i in range(dimensions[0]):
    for j in range(dimensions[1]):
      tuple_state = (i,j)
      act_acts = act[tuple_state]
      # print('acts_acts:')
      # print(act_acts)
      prev_acts = prev[tuple_state]
      # print('prev_acts:')
      # print(prev_acts)
      act_action = max(act_acts, key=act_acts.get)
      prev_action = max(prev_acts, key=prev_acts.get)
      if not act_action==prev_action:
        converged=False
        break
    if not converged:
      break
  return converged

def check_values_convergence(act,prev,dimensions):
  # print('check convergence')
  converged = True
  for i in range(dimensions[0]):
    for j in range(dimensions[1]):
      tuple_state = (i,j)
      act_vals = list(act[tuple_state].values())
      # print('acts_vals:')
      # print(act_vals)
      prev_vals = list(prev[tuple_state].values())
      # print('prev_vals:')
      # print(prev_vals)
      for k in range(len(act_vals)):
        act_val = act_vals[k]
        prev_val = prev_vals[k]
        diff = abs(act_val-prev_val)
        if diff>0.1:
          converged=False
          break
      if not converged:
        break  
    if not converged:
      break
  return converged

def create_taxi_qtable(np_form,acts,dimensions):
  acts_dict=dict.fromkeys(acts,0)
  prev_dict=dict.fromkeys(acts,-np.inf)
  list_form = np_form.tolist()
  prev = copy.deepcopy(list_form)
  for i in range(dimensions[0]):
    for j in range(dimensions[1]):
      for k in range(dimensions[2]):
        list_form[i][j][k]=copy.deepcopy(acts_dict)
        prev[i][j][k]=copy.deepcopy(prev_dict)
  act = np.array(list_form)
  prev = np.array(prev)
  return act,prev



x=Cell(0,False)
wl = Cell(0,False,['left'])
wr = Cell(0,False,['right'])
r=Cell('r',True)
g=Cell('g',True)
b=Cell('b',True,['left'])
y=Cell('y',True,['right'])
taxi_board=[
    [r,wr,wl,x,g],
    [x,wr,wl,x,x],
    [x,x,x,x,x],
    [wr,wl,wr,wl,x],
    [y,wl,wr,b,x]
]
possible_stops = ['r','g','b','y']
taxi_dimensions=[len(taxi_board),len(taxi_board[0]),len(possible_stops)+1]
taxi_actions = ['left','right','up','down','pick','drop']


def taxi_check_policy_convergence(act,prev,dimensions):
  converged = True
  for i in range(dimensions[0]):
    for j in range(dimensions[1]):
      # for z in range(dimensions[2]):
      for z in range(2):
        tuple_state = (i,j,z)
        act_acts = act[tuple_state]
        # print('acts_acts:')
        # print(act_acts)
        prev_acts = prev[tuple_state]
        # print('prev_acts:')
        # print(prev_acts)
        act_action = max(act_acts, key=act_acts.get)
        prev_action = max(prev_acts, key=prev_acts.get)
        prev_action_val = prev_acts[prev_action]
        if not act_action==prev_action or -np.inf in prev_acts.values():
          converged=False
          break
      if not converged:
        break
    if not converged:
        break
  return converged

def taxi_check_values_convergence(act,prev,dimensions):
  converged = True
  for i in range(dimensions[0]):
    for j in range(dimensions[1]):
      # for z in range(dimensions[2]):
      for z in range(2):
        tuple_state = (i,j,z)
        act_vals = list(act[tuple_state].values())
        prev_vals = list(prev[tuple_state].values())
        for k in range(len(act_vals)):
          act_val = act_vals[k]
          prev_val = prev_vals[k]
          diff = abs(act_val-prev_val)
          if diff>0.1:
            converged=False
            break
          if not converged:
            break  
      if not converged:
        break
    if not converged:
        break
  return converged

alpha=0.3
gamma=0.9
epsilon=0.9
board = copy.deepcopy(taxi_board)
acts = copy.deepcopy(taxi_actions)
# board, dimensions, initial_state,actions,reward_function
dimensions=[len(board),len(board[0]),len(board[1])]
taxi_qtable=np.zeros((len(taxi_board),len(taxi_board[0]),len(possible_stops)+1))
# dimensions=[len(board),len(board[0]),2]
# taxi_qtable=np.zeros((len(taxi_board),len(taxi_board[0]),2))
taxi_qtable,taxi_prev_qtable=create_taxi_qtable(taxi_qtable,taxi_actions,taxi_dimensions)
env = TaxiEnvironment(board,dimensions,acts,possible_stops)
agent = Agent(acts)
learn = Learner(agent,env,taxi_qtable,taxi_prev_qtable,taxi_check_policy_convergence,alpha,gamma,epsilon,25000)
learn.run_til_convergence()
qtable=learn.qtable