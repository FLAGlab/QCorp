#definiciones -- se incluye segmentos assignment anteriores
import numpy as np
import math
import copy


#Definición de clase Tablero, atendiendo el requerimiento "Las recompensas de cada casilla de la cuadrícula estan definidas dentro de la definición de board."
class Board:
    #definición de atributo tablero creado de acuerdo con dimensiones y configuración que se envía por parametro 
    def __init__(self, w, h, conf):
        #definición de atributo dimensión y asignación de valores de acuerdo con dimensiones que se envían por parametro 
        self.width = w
        self.height = h
        #definición de atributo tablero creado de acuerdo con dimensiones que se envían por parametro 
        self.board = [[0 for i in range(self.width)] for j in range(self.height)]
        for row in conf:
          self.board[row[0]][row[1]] = row[2]

    def getBoard(self):
        return self.board