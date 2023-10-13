import numpy as np
import matplotlib.pyplot as plt
from environments import TaxiEnv, taxi_env

class QTaxi:
    def __init__(self, step_size: float, discount: float, epsilon: float, env: TaxiEnv) -> None:
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.env = env
        # This is not efficient because it stores non useful states.
        # Initialize Q as a Numpy array of 0s for each state.
        self.Q = np.zeros((self.env.height, self.env.width, 2, len(self.env.actions)))
        self.last_state = self.env.current_state

    # Off course I assume there is at least one action to be made.
    def argmaxA_Q(self, state, has_passenger):
        x, y = state
        max_Q = float("-inf")
        maxs = []
        for action in self.env.actions:
            if self.Q[x][y][has_passenger][action] > max_Q:
                maxs = [action]
                max_Q = self.Q[x][y][has_passenger][action]
            elif self.Q[x][y][has_passenger][action] == max_Q:
                maxs.append(action)
        return np.random.choice(maxs)

    # Executes one step of the episode. Returns whether or not S' (the new state) is terminal.
    # Uses episode_num to make alpha converge.
    def episode_step(self, episode_num, converges):
        # "Old" state.
        x, y = self.last_state
        old_passenger_state = self.env._has_passenger

        # Choose A from S using policy derived from Q (e.g., E-greedy).
        probability = np.random.random()
        if probability < self.epsilon:
            action = np.random.choice(self.env.actions)
        else:
            action = self.argmaxA_Q(self.last_state, old_passenger_state)
        
        # Take action A, observe R, S'.
        (new_state, reward, terminal, has_passenger) = self.env.do_action(action)
        nx, ny = new_state

        old_Q = self.Q[x][y][has_passenger][action]
        # Update Q.
        # * (1/episode_num) would make the series converge with certainty.
        self.Q[x][y][old_passenger_state][action] = self.Q[x][y][old_passenger_state][action] \
            + self.step_size \
                *(reward + self.discount*self.Q[nx][ny][has_passenger][self.argmaxA_Q((nx, ny), has_passenger)] \
                  - self.Q[x][y][old_passenger_state][action])

        # Determines if the Qvalues converge.
        new_Q = self.Q[x][y][has_passenger][action]
        if not (abs(new_Q) * 0.99 < abs(old_Q) < abs(new_Q) * 1.01):
            converges = converges & False

        # Update S <- S'.        
        self.last_state = new_state

        return (terminal, converges)
        
    def control(self):
        episode_num = 0
        converges = False
        while not converges:
            terminal = False
            converges = True
            while(not terminal):
                terminal, converges = self.episode_step(episode_num+1, converges)
            self.last_state = self.env.current_state
            episode_num += 1
        return episode_num

    def generate_policy(self, has_passenger):
        policy = np.full((self.env.height, self.env.width), len(self.env.actions))
        for x in range(self.env.height):
            for y in range(self.env.width):
                if (self.Q[x][y] == [0 for _ in self.env.actions]).all():
                    continue
                best_actions = []
                best_Q = float("-inf")
                for action in self.env.actions:
                    if self.Q[x][y][has_passenger][action] > best_Q:
                        best_actions = [action]
                        best_Q = self.Q[x][y][has_passenger][action]
                    elif self.Q[x][y][has_passenger][action] == best_Q:
                        best_actions.append(action)
                policy[x][y] = np.random.choice(best_actions)
        return policy
    
    def plot_scenario(self, has_passenger):
        plt.rcParams.update({'font.size': 7})
        policy = self.generate_policy(has_passenger)
        _, axes = plt.subplots()
        axes.set_title(f"Has passenger == {'True' if has_passenger else 'False'}")
        axes.set_aspect('equal')
        axes.set_xlim(0, self.env.width)
        axes.set_ylim(self.env.height, 0)
        for y in range(self.env.width):
            for x in range(self.env.height):
                cell_policy = policy[x][y]
                text = ""
                if cell_policy == len(self.env.actions):
                    color = 'dimgray'
                    cell_qvalue = float("-inf")
                elif (x, y) == self.env._passenger_pickup:
                    color = 'greenyellow'
                    cell_qvalue = round(self.Q[x][y][has_passenger][cell_policy], 2)
                    text = f"{policy_to_string(cell_policy)}\n{cell_qvalue}"
                elif (x, y) == self.env._passenger_dropoff:
                    color = 'pink'
                    cell_qvalue = round(self.Q[x][y][has_passenger][cell_policy], 2)
                    text = f"{policy_to_string(cell_policy)}\n{cell_qvalue}"
                else:
                    color = 'white'
                    cell_qvalue = round(self.Q[x][y][has_passenger][cell_policy], 2)
                    text = f"{policy_to_string(cell_policy)}\n{cell_qvalue}"
                axes.add_patch(plt.Rectangle((y, x), 1, 1, facecolor=color))
                axes.text(y + 0.5, x + 0.5, text, ha='center', va='center')
        plt.show()

def policy_to_string(policy):
    if policy == 0:
        return "∧"
    elif policy == 1:
        return "v"
    elif policy == 2:
        return "<"
    elif policy == 3:
        return ">"
    elif policy == 4:
        return "D"
    elif policy == 5:
        return "P"
    else:
        return ""
    
agent = QTaxi(0.1, 0.9, 0.5, taxi_env)
print(f"Episode number: {agent.control()}")
agent.plot_scenario(0)
agent.plot_scenario(1)