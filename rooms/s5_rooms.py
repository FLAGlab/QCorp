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

class Environment:
  def __init__(self, board, dimensions,actions,reward_function):
    self.board=board
    self.dimensions=dimensions
    self.is_terminal=False
    self.actions=actions
    self.reward_function = reward_function
    self.reset()

  def get_dimensions(self):
    return self.dimensions
  
  def get_current_state(self):
    return self.current_state 

  def do_action(self,action):
    return self.do_action_final_action(action)

  def do_action_final_action(self,action):
    reward,self.current_state, self.is_terminal = self.reward_function(self.current_state, action, self.board, self.dimensions)
    return [reward,self.current_state]

  def reset(self):
    self.is_terminal=False
    y=random.randrange(self.dimensions[0])
    x=random.randrange(self.dimensions[1])
    while self.board[y][x] == '*':
      y=random.randrange(self.dimensions[0])
      x=random.randrange(self.dimensions[1])
    self.current_state=[y,x]

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

def create_qtable(np_form,acts):
  acts_dict=dict.fromkeys(acts,0)
  prev_dict=dict.fromkeys(acts,-np.inf)
  list_form = np_form.tolist()
  prev = copy.deepcopy(list_form)
  for i in range(len(list_form)):
    for j in range(len(list_form[0])):
      list_form[i][j]=copy.deepcopy(acts_dict)
      prev[i][j]=copy.deepcopy(prev_dict)
  act = np.array(list_form)
  prev = np.array(prev)
  return act,prev


x=Cell(0,False)
y=Cell(1,True)
wd = Cell(0,False,['down'])
wu = Cell(0,False,['up'])
wl = Cell(0,False,['left'])
wr = Cell(0,False,['right'])
wur = Cell(0,False,['up','right'])
wul = Cell(0,False,['up','left'])
wdr = Cell(0,False,['down','right'])
wdl = Cell(0,False,['down','left'])

maze_board=[
    [x,x,y,x,wr,wl,x,x,x,x],
    [x,x,x,x,wr,wl,x,x,x,x],
    [x,x,x,x,x,x,x,x,x,x],
    [x,x,x,x,wr,wl,x,x,x,x],
    [wd,wd,x,wd,wdr,wdl,wd,x,wd,wd],
    [wu,wu,x,wu,wur,wul,wu,x,wu,wu],
    [x,x,x,x,wr,wl,x,x,x,x],
    [x,x,x,x,x,x,x,x,x,x],
    [x,x,x,x,wr,wl,x,x,x,x],
    [x,x,x,x,wr,wl,x,x,x,x]
]
maze_acts_dict={
    'left':0,
    'right':0,
    'up':0,
    'down':0
}
maze_actions=['left','right','up','down']


def maze_reward_function(current_state, action, board, dimensions):
    # print('llamando maze_reward_function')
    actx,acty,left,right,down,up = get_area(current_state)
    reward = 0 
    terminal=False
    if action in board[acty][actx].walls:
      return [0,current_state,terminal]
    if action=='left' and left>=0:
        current_state[1]-=1
    elif action=='right' and right<dimensions[1]:
        current_state[1]+=1
    elif action=='up':
      if board[current_state[0]][current_state[1]].is_terminal:
        terminal=True
        reward=board[current_state[0]][current_state[1]].get_value()
      elif up>=0:
        current_state[0]-=1
    elif action=='down' and down<dimensions[0]:
        current_state[0]+=1
    # print('info')
    # print(f'Reward:{reward} Current state:{current_state} is_terminal:{terminal}')
    return [reward,current_state,terminal]

alpha=0.1
gamma=0.9
epsilon=0.9
board = copy.deepcopy(maze_board)
acts = copy.deepcopy(maze_actions)
dimensions=[len(board),len(board[0])]
# board, dimensions, initial_state,actions,reward_function
maze_qtable = np.zeros((len(maze_board),len(maze_board[0])))
maze_qtable, maze_prev_qtable = create_qtable(maze_qtable,maze_actions)
env = Environment(board,dimensions,acts,maze_reward_function)
agent = Agent(acts)
learn = Learner(agent,env,maze_qtable,maze_prev_qtable,check_policy_convergence,alpha,gamma,epsilon)
learn.run_til_convergence()
qtable=learn.qtable