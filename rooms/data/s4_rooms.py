import numpy as np
import matplotlib.pyplot as plt

class QLearning:

    def __init__(self, 
                 env, agent, 
                 epsilon, alpha, gamma,
                 decrease_alpha = 0.1, exploration_decreasing_decay = 0.01,
                 num_episodes_batch = 100
                 ) -> None: 
        """
        Inicializa los valores del learner
        """
        # Environment and agent
        self.env = env 
        self.agent = agent 

        # Hyper parameters
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.num_episodes_batch = num_episodes_batch

        # Decrease (alpha and epsilon)
        self.decrease_alpha = decrease_alpha
        self.exploration_decreasing_decay = exploration_decreasing_decay
        
        # Q table and actions
        self.possible_actions = agent.actions
        self.Q = {(s, a): 0 for s in env.states for a in self.possible_actions}


    def Q_to_matrix(self, Q):
        """
        Retorna una matriz con los Q-valores de cada par (s,a).
        """
        Q_mat = np.zeros((len(self.env.states), len(self.possible_actions)))
        for i, s in enumerate(self.env.states):
            for j, a in enumerate(self.possible_actions):
                Q_mat[i,j] = Q[(s,a)]
        return Q_mat

    def choose_action(self, state):
        """
        Elige una acción a tomar desde un estado dado siguiento epsilon-greedy.
        Si el estado es final, retorna None.
        """
        if state == 'final': return None 

        option = np.random.choice(["explore", "explote"], p=[self.epsilon, 1-self.epsilon])
        if option == "explore": action = np.random.choice(self.possible_actions)
        else: 
            action = self.possible_actions[0]
            for a in self.possible_actions:
                if self.Q[(state, a)] > self.Q[(state, action)]: 
                    action = a 
            max_actions = [ac for ac in self.possible_actions if self.Q[(state, action)] == self.Q[(state, ac)]]
            action = np.random.choice(max_actions)
        
        return action

    def action_function(self, state1, action1, reward, state2):
        """
        Actualiza los valores de la Q-tabla
        """
        if state2 == "final": q = 0
        else: q = np.max([ self.Q[(state2, a)] for a in self.possible_actions ]) 

        ans = (1-self.alpha)*self.Q[(state1,action1)]+(self.alpha*(reward+self.gamma*q))
        return ans 
    
    def run_episode(self):
        """
        Ejecuta un episodio. Finaliza cuando encuentra un estado terminal. Retorna la recompensa total del episodio.
        """
        # Starting state
        start_states = self.env.start()
        state1 = start_states[np.random.choice(range(len(start_states)))]
        total_reward = 0
        steps = 0
        while True:
            steps +=1
            # Stopping criterion
            if state1 == 'final':  break 

            action1 =  self.choose_action(state1)
            reward, state2 = self.agent.do_action(state1, action1)

            total_reward += reward
            self.Q[(state1, action1)] = self.action_function(state1, action1, reward, state2)

            state1 = state2  
        
        return total_reward, steps

    def qlearning(self):
        episode = 0
        initial_epsilon = self.epsilon
        initial_alpha = self.alpha
        last_reward_Nepisodes = -100

        epsilons, alphas, rewards, steps = [], [], [], []

        while True: 

            reward_Nepisodes, stepsNepisodes = [], []
            for _ in range(self.num_episodes_batch):
                episode += 1
                total_reward, step = self.run_episode()
                reward_Nepisodes.append(total_reward)
                stepsNepisodes.append(step)

                if episode %10 == 0:
                    print(f'--> Episode {episode}. Reward: {total_reward}. Steps: {step} Epsilon: {round(self.epsilon,4)}. Alpha: {round(self.alpha,4)}')
            
            epsilons.append(self.epsilon)
            alphas.append(self.alpha)
            rewards.append(np.mean(reward_Nepisodes))
            steps.append(np.mean(stepsNepisodes))

            # Cada 100 episodios se recalcula el valor de epsilon. De esta forma va a disminuir las veces que explora.
            self.epsilon = initial_epsilon*np.exp(-self.exploration_decreasing_decay*int(episode/100))
            self.alpha = initial_alpha*np.exp(-self.exploration_decreasing_decay*int(episode/100))

            # Si la diferencia entre la recompensa media de los anteriores 100 episodios y de los últimos es pequeña, 
            # Se terminan las iteraciones
            x = np.abs(np.mean(reward_Nepisodes) - last_reward_Nepisodes)
            print("\n", x, "\n")
            if  x < 0.001 and self.epsilon < 0.1: break
             
            last_reward_Nepisodes = np.mean(reward_Nepisodes)  


        return  epsilons, alphas, rewards, steps

import numpy as np

class LaberintoCuartosAgente:

    def __init__(self) -> None:
        self.actions = ['left', 'right', 'up', 'down', 'exit']
        self.state = (0,0)

    def getPossibleActions(self, state) -> list:
        """
        Dado un estado, retorna una lista de posibles acciones
        """
        possible_actions = []

        # Llegó a la salida
        if state == (0,2): possible_actions = ['exit']

        else:
            # Puede ir a la derecha
            if (state[1] < 9 and state[1] != 4) or (
                state[1] == 4 and state[0] in (2,7)):
                possible_actions.append('right')
            
            # Puede ir a la izquierda
            if (state[1] > 0 and state[1] != 5) or (
                state[1] == 5 and state[0] in (2,7)):
                possible_actions.append('left')
            
            # Puede ir a arriba
            if (state[0] > 0 and state[0] != 5) or (
                state[0] == 5 and state[1] in (2,7)):
                possible_actions.append('up')

            # Puede ir a abajo
            if (state[0] < 9 and state[0] != 4) or (
                state[0] == 4 and state[1] in (2,7)):
                possible_actions.append('down')

        return possible_actions
    
    def do_action(self, state, action):
        """
        Dado un estado y una acción, retorna la recompensa y el estado de finalización
        """
        row, column = state 
        possible_actions = self.getPossibleActions(state)
        other_possible_actions = [a for a in possible_actions if a != action]

        # A la salida hay una recompensa de 1
        if action == 'exit' and state == (0,2): return 1, 'final'
        
        if not action in possible_actions: 
            action_chosen = np.random.choice([*['stay'], *other_possible_actions],
                                        p = [*[0.7], *[0.3/len(other_possible_actions)]*len(other_possible_actions)])
        else: 
            action_chosen = np.random.choice([*[action], *other_possible_actions],
                                        p = [*[0.7], *[0.3/len(other_possible_actions)]*len(other_possible_actions)])
        
        if action_chosen == 'stay': new_state = state
        elif action_chosen == 'right': new_state = (row, column+1)
        elif action_chosen == 'left': new_state = (row, column-1)
        elif action_chosen == 'up': new_state = (row-1, column)
        else: new_state = (row+1, column)

        self.state = new_state

        return -0.001, new_state

class LaberintoCuartosAmbiente:

    def __init__(self) -> None:
        self.states = [(i,j) for i in range(10) for j in range(10)]

    def start(self) -> list:
        return self.states

    def end(self) -> list:
        return ['final']


env = LaberintoCuartosAmbiente()
agent = LaberintoCuartosAgente()
learner = QLearning( env = env, agent = agent,  
                            epsilon = 0.8, alpha = 0.4, gamma = 0.9,
                            decrease_alpha = 0.01, exploration_decreasing_decay = 0.01,
                            num_episodes_batch = 200)
epsilons, alphas, rewards, steps = learner.qlearning()