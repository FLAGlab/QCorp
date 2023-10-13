import numpy as np

class TaxiAmbiente:

    def __init__(self) -> None:
        self.states = [(i,j,k) for i in range(5) for j in range(5) for k in (0,1)]
        
        self.paradas = [(0,0), (0,4), (4,0), (4,4)]
        self.pasajero_inicial = (0,0,0) # Para lograr la convergencia, siempre se parte del mismo punto y llega al mismo punto np.random.choice(self.paradas) 
        self.pasajero_destino = (4,0,1) # np.random.choice([s for s in self.paradas if s != self.pasajero_inicial])

        self.final_states = ['final']

    def start(self) -> list:
        return [(i,j,0) for i in range(5) for j in range(5)]

    def stop(self) -> list:
        return self.final_states