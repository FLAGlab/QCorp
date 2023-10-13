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
