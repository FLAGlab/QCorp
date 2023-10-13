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
