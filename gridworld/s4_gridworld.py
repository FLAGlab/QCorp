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

class EnvironmentGridWorld:

    def __init__(self) -> None:
        self.board = self.crear_ejemplo_gw()
        dimension = len(self.board)
        self.dimension = dimension
        self.states = [(i,j) for i in range(dimension) for j in range(dimension)]

        for i in range(dimension):
            for j in range(dimension):
                if self.board[i,j] == -3: self.states.remove((i,j))
    
    def crear_ejemplo_gw(self):
        # Se crea el board con todos los valores en 0
        dimension = 10
        board = np.zeros((dimension, dimension))
        # Casillas prohibidas
        casillas_prohibidas = [(2,1), (2,2), (2,3), (2,4), (2,6), (2,7), (2,8),
                            (3,4), (4,4), (5,4), (6,4), (7,4)]

        for (r,c) in casillas_prohibidas:
            board[r,c] = -3

        # Casillas trampa
        casillas_trampa = [(4,5), (7,5), (7,6)]

        for (r,c) in casillas_trampa:
            board[r,c] = -1

        # Casilla finalización
        casillas_finalizacion = (5,5)
        board[casillas_finalizacion] = 1

        return board

    def start(self):
        "Retorna una lista de estados iniciales"
        return [(0,0)]
    
    def end(self):
        "Retorna una lista de estados finales"
        return ["final"]
        

import numpy as np

class AgentGridWorld:
    
    def __init__(self, env):
        self.board = env.board
        self.dimension = env.dimension
        
        self.state = (0,0)
        self.actions = ['exit', 'up', 'down', 'right', 'left']
    
    def get_possible_actions(self, state):
        """
        Dado un estado se retorna una lista de las acciones factibles para el estado.
        Es decir, acciones que puedan modificar el estado. No puede salirse del tablero, no puede salir sin estar en el estado final.
        """
        row, column = state

        if self.board[state] == -3: return []

        possible_actions = []
        if column < self.dimension-1 and self.board[row, column + 1] != -3:
            possible_actions.append('right')
        if column > 0 and self.board[row, column - 1] != -3:
            possible_actions.append('left')
        if row < self.dimension-1 and self.board[row+1, column] != -3:
            possible_actions.append('down')
        if row > 0 and self.board[row-1, column] != -3:
            possible_actions.append('up')
        if self.board[state] == 1:
            possible_actions.append('exit')

        return possible_actions
        
    def do_action(self, state, action):
        row, column = state
        possible_actions = self.get_possible_actions(state)
        
        if action == 'exit' and state == (5,5): return 1, 'final'

        other_possible_actions = [a for a in possible_actions if a != action]

        # Se considera un ruido de 0.1
        # (con probabilidad 0.9 se toma la acción que se eligió, con 0.1 de probabilidad se toma otra acción)
        if not action in possible_actions: 
            action_chosen = np.random.choice([*['stay'], *other_possible_actions],
                                        p = [*[0.8], *[0.2/len(other_possible_actions)]*len(other_possible_actions)])
        else: 
            action_chosen = np.random.choice([*[action], *other_possible_actions],
                                        p = [*[0.8], *[0.2/len(other_possible_actions)]*len(other_possible_actions)])

        # Se calcula el nuevo estado
        if action_chosen == 'stay': new_state = (row, column)
        elif action_chosen == 'right': new_state = (row, column+1)
        elif action_chosen == 'left': new_state = (row, column-1)
        elif action_chosen == 'down': new_state = (row+1, column)
        else: new_state = (row-1, column)
        
        # El valor de 1 se da cuando se tome la acción 'exit' desde la casilla de finalización y no cuando se llega.
        if self.board[new_state] == 1: reward = 0
        else: reward = self.board[new_state]

        # Se actualiza el estado del agente
        self.state = new_state

        return reward, new_state

env = EnvironmentGridWorld()
agent = AgentGridWorld(env)
learner = QLearning( env = env, agent = agent,  
                            epsilon = 0.8, alpha = 0.4, gamma = 0.9,
                            decrease_alpha = 0.01, exploration_decreasing_decay = 0.01,
                            num_episodes_batch = 500)
epsilons, alphas, rewards, steps = learner.qlearning()