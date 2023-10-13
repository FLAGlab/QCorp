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
        