from ambiente_taxi import TaxiAmbiente

class TaxiAgente:

    def __init__(self, ambiente_taxi) -> None:
        self.ambiente_taxi: TaxiAmbiente = ambiente_taxi
        self.actions = ['left', 'right', 'up', 'down', 'recogerPasajero', 'dejarPasajero']
        self.state = (0,0,0)

    def getPossibleActions(self, state) -> list:
        """
        Dado un estado, retorna una lista de posibles acciones
        """
        possible_actions = []

        # Si el taxi se encuentra en el destino del pasajero y ya lo recogió, puede dejarlo y finaliza el episodio. 
        # De lo contrario no.
        if state == self.ambiente_taxi.pasajero_destino: 
            possible_actions = ['dejarPasajero']

        else:
            # Puede ir a la derecha
            if (state[1] == 3) or (state[1] == 0 and state[0] < 3) or (
                state[1] == 1 and state[0] > 1) or (state[1] == 2 and state[0] < 3):
                possible_actions.append('right')
            
            # Puede ir a la izquierda
            if (state[1] == 4) or (state[1] == 1 and state[0] < 3) or (
                state[1] == 2 and state[0] > 1) or (state[1] == 3 and state[0] < 3):
                possible_actions.append('left')
            
            # Puede ir a arriba
            if state[0] > 0:
                possible_actions.append('up')

            # Puede ir a abajo
            if state[0] < 4:
                possible_actions.append('down')

            # Puede recoger un pasajero
            if state == self.ambiente_taxi.pasajero_inicial:
                possible_actions.append('recogerPasajero')

        return possible_actions
    
    def do_action(self, state, action):
        """
        Dado un estado y una acción, retorna la recompensa y el estado de finalización
        """

        row, column, occupied = state 
        possible_actions = self.getPossibleActions(state)

        if not action in possible_actions: 
            # Si intenta recoger o dejar un pasajero y no puede, tiene recompensa -10
            if action in ['recogerPasajero', 'dejarPasajero']: return -10, state 
            
            # Si intenta hacer un movimiento prohibido, que no sea dejar o recoger, se da recompensa 0
            else: return 0, state 

        # Si deja al pasajero hay una recompensa de 5 y se sale del juego
        if action == 'dejarPasajero': return 5, 'final'

        # Si recoge al pasajero hay una recompensa de 1 y se queda en la misma posición
        if action == 'recogerPasajero': return 1, (row, column, 1)

        if action == 'right': new_state =  (row, column+1, occupied)
        if action == 'left': new_state = (row, column-1, occupied)
        if action == 'up': new_state = (row - 1, column, occupied)
        if action == 'down': new_state = (row+1, column, occupied)

        self.state = new_state 
        # if new_state in [self.ambiente_taxi.pasajero_inicial,self.ambiente_taxi.pasajero_destino]: print(self.state)

        return 0, new_state