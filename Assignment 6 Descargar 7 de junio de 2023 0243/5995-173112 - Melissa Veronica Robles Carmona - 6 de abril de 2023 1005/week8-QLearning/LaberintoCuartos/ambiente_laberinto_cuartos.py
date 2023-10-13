class LaberintoCuartosAmbiente:

    def __init__(self) -> None:
        self.states = [(i,j) for i in range(10) for j in range(10)]

    def start(self) -> list:
        return self.states

    def end(self) -> list:
        return ['final']


