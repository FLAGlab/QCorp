import random
import matplotlib.pyplot as plt
import copy


board = [
  [' ',' ', '1',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ',' ',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  ['*','*', ' ','*','*','*','*','*',' ','*','*'],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ',' ',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
  [' ',' ', ' ',' ',' ','*',' ',' ',' ',' ',' '],
]

class GridWorld:
  def __init__(self, board=None):
    self.board = board or default_board
    self.dimensions = (len(self.board),len(self.board[0]))
    self.current_state = (0,0)
    self.actions = ['F','↑','↓','←','→']
    self.terminal_actions = ['F']

  def get_current_state(self):
    return self.current_state

  def get_posible_actions(self, coordinates):
    # If the agent is in the border then it can't move out of the board
    # The agent can't move to a cell that has an obstacle
    max_row, max_column = self.dimensions
    lower_row = 0
    upper_row = max_row - 1
    leftmost_col = 0
    rightmost_col = max_column - 1

    if coordinates=='finished':
      return None
    
    curr_row, curr_col = coordinates

    possible_actions = []


    if self.is_terminal(coordinates):
      # If it is a terminal state then return the finish action
      return ['F']
    if curr_row > lower_row:
      possible_actions.append('↑')
    if curr_row < upper_row:
      possible_actions.append('↓')
    if curr_col > leftmost_col:
      possible_actions.append('←')
    if curr_col < rightmost_col:
      possible_actions.append('→')

    return possible_actions
  
  def get_start_state(self):
    return (0,0)
  
  def is_valid_action(self, prev_state, state):
    return  not (state[0] < 0 or state[1] < 0 or  self.board[state[0]][state[1]] == '*')

  def get_reward(self, prev_state, new_state, action):
    return int(self.board[prev_state[0]][prev_state[1]])

  def do_action(self, action):
    curr_row, curr_col = self.get_current_state()
    new_position = None

    # Get the new position
    if action == 'F':
      new_position = (curr_row, curr_col)
    elif action == '↑':
       new_position = (curr_row-1, curr_col)
    elif action == '↓':
      new_position = (curr_row+1, curr_col)
    elif action == '←':
      new_position = (curr_row, curr_col-1)
    elif action == '→':
      new_position = (curr_row, curr_col+1)

    # Check if the new position is valid
    try:
      if not self.is_valid_action((curr_row, curr_col),new_position):
        new_position = (curr_row, curr_col)
    except:
      new_position = (curr_row, curr_col)

    # Calculate the reward as the current state reward
    try:
      reward = self.get_reward((curr_row, curr_col),new_position,action)
    except:
      reward = 0

    # When the finish action was send set the state of the game as finished
    try:
      if self.is_terminal(new_position) and action in self.terminal_actions:
        new_position = 'finished'
    except:
      print('error pos', new_position)

    self.current_state = new_position
    return (reward, new_position)

  def set_current_state(self, new_state):
    row, column = new_state
    self.current_state = (row, column)

  def is_game_over(self, state):
    return state == 'finished'

  def reset(self):
    new_state = None
    while True:
      row = random.randint(0, self.dimensions[0] -1)
      col = random.randint(0, self.dimensions[1] -1)
      if self.board[row][col] != '*':
        self.current_state = (row,col)
        return (row,col)
    

  def is_terminal(self, state):
    curr_row, curr_col = state
    return self.board[curr_row][curr_col] == '1'
  

class Q_Learning():
  def __init__(self, env, discount=0.9, epsilon=0.1, alpha=0.25):
    self.env = env()
    self.discount = discount
    self.epsilon = epsilon
    self.alpha = alpha
    # Matrix to save the value of the state-action pair
    self.Q = {}

  def get_action(self, state):
    return self.calculate_best_action(state)

  # An e-policy that takes the greedy action with 1-e probability and random otherwise
  def choose_action(self, state, random_act = False):
    # Check the possible actions
    possibilities = self.env.get_posible_actions(state)
    # If there are no actions for the state return None
    if not possibilities:
      return None
    # Take a random action with epsilon probability
    if random.random() < self.epsilon or random_act:
      return random.choice(possibilities)
    # Take the greedy action
    return self.calculate_best_action(state)

  # Simulates an episode
  def take_a_step(self):
    curr_state=self.env.get_current_state()
    # Reset the game when the agent reaches the end
    if self.env.is_game_over(curr_state):
      self.env.reset()
      curr_state=self.env.get_current_state()
    # Get the next action
    next_action = None
    while not next_action:
      next_action = self.choose_action(curr_state)
    # Execute the action
    reward, new_position = self.env.do_action(next_action)
    return (curr_state, next_action, reward, new_position)

  def get_value(self, state, action):
    if self.env.is_game_over(state):
      return 0
    return self.Q.get((state,action),0)

  # For a given state choose the action with higher current q value
  def calculate_best_action(self, state):
    best_actions = []
    new_policy = None
    maxVal =  float('-inf')
    possibilities = self.env.get_posible_actions(state)
    if not possibilities:
      return None
    
    for action in possibilities:
      val = self.get_value(state, action)
      # Update the max possible q value
      if val > maxVal:
        maxVal = val
        best_actions = [action]
      elif val == maxVal:
        best_actions.append(action)

    # Break ties randomly
    new_policy = random.choice(best_actions)
    return new_policy
  
  def action_function(self, state, action, reward, new_state):
    curr_value = self.get_value(state, action)
    new_state_action = self.get_action(new_state) # Best action for the state
    next_value = self.get_value(new_state, new_state_action)
    self.Q[(state,action)] = (1-self.alpha)*curr_value + self.alpha*(reward + self.discount*next_value)

  # Simulates an episode, calculates the expected value for every state-action pair, and improves the policy
  def run_an_step(self, logs=False):
    step = self.take_a_step()
    if logs:
      print('step:',step)

    state, action, reward, new_state = step
    # Update the estimate of the state value
    self.action_function(state, action, reward, new_state)

  # Utility function to plot the Q values and the policy
  def show_values_on_the_board(self):
    state_values = copy.deepcopy(self.env.board)
    for row in range(len(state_values)):
      for column in range(len(state_values[row])):
        state_values[row][column] = self.get_value((row, column), self.get_action((row, column)))
    return state_values

  # Plots the current greedy policy and the expected values of each state with such policy
  def plot_policy(self):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(self.show_values_on_the_board())
    rows = self.env.dimensions[1]
    columns = self.env.dimensions[0]
    ax.set_xticks(list(range(columns)))
    ax.set_yticks(list(range(rows)))
    states = [(i,j) for j in range(columns) for i in range(rows)]
    for s in states:
      policy = None if self.env.board[s[0]][s[1]] == '*' else self.get_action(s)
      ax.annotate(f'${policy}: ${self.get_value(s, policy):.1f}' if policy else '*', xy=(s[1], s[0]), xycoords='data', color='white' if policy else 'black',va='center', ha='center', fontsize=12 if policy else 20)

class Laberinto(GridWorld):
  def __init__(self):
    super().__init__(board)

def main():
    q_learning_agent = Q_Learning(Laberinto)
    
    for _ in range(400000):
        q_learning_agent.run_an_step()
    #q_learning_agent.plot_policy()