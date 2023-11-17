import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import random
import operator
pd.options.display.max_rows = None

#definimos la clase board
class board():
    def __init__(self,n=10):
        self.n=n#dimensiones de la cuadricula nxn
        self.current_state=(0,0)#posicion inicial del agente
        self.reward=0#recompensa obtenida por el agente
        self.grid=np.zeros((11,11)) #almacenamos el gridworld como una matriz 3x7
        self.__values=np.zeros((11,11))#creamos una matriz con las mismas dimensiones que el grod world
        self.grid[5,0:2]=np.nan#definimos aqui 
        self.grid[5,3:8]=np.nan#las paredes de los
        self.grid[5,9:11]=np.nan#3 cuartos
        self.grid[0:2,5]=np.nan#definimos aqui 
        self.grid[3:8,5]=np.nan#las paredes de los
        self.grid[9:11,5]=np.nan#3 cuartos
        #para almacenar los valores de los estados(entiendase valores como los que se calculan de la ecuación de 
        #Bellman)
        self.__values[0,2]=1#Asignamos una recompensa de 1 a la salida del laberinto
        
        
        
        
        
        self._is_term=np.zeros((11,11))
        self._is_term[0,2]=1 #asignamos como casilla terminal la salida del laberinto
        
    def get_current_state(self):
        return self.current_state
    def get_posible_actions(self, estado_actual):
        fila, columna=estado_actual #desempacamos la fila y la columna
        n_filas,n_columnas=self.grid.shape
        #del estado actual
        acciones=[]
        n=self.n
        if self.is_terminal(estado_actual)==True:
            acciones.append("Salir")
            return acciones
        #verificar si up es posible:
        if (fila!=0) and (np.isnan(self.grid[fila-1,columna])==False):#se puede mover hacia arriba
            #si no esta en la fila cero y ademas en la casilla superior no hay obstaculo
            acciones.append("up")
        #verificar si down es posible
        if (fila!=n_filas-1) and (np.isnan(self.grid[fila+1,columna])==False):#se puede mover hacia abajo
            #si no esta en la ultima fila y ademas la casilla de abajo no es obstaculo
            acciones.append("down")
        #verificar si right es posible
        if (columna!=n_columnas-1) and (np.isnan(self.grid[fila,columna+1])==False):#se puede mover hacia la
            #derecha si no esta en la ultima columna y ademas la casilla de la derecha no es obstaculo
            acciones.append("right")
        if (columna!=0) and (np.isnan(self.grid[fila,columna-1])==False):#se puede mover hacia la izquierda
            #si no esta en la primera coulumna y ademas la casilla de la izquiera no es un obstaculo
            acciones.append("left")
        return acciones
    
    def do_action(self, action):
        fila_actual, columna_actual=self.current_state
        acciones_posibles=self.get_posible_actions((fila_actual,columna_actual))
        if action not in acciones_posibles:
            return (0,self.current_state)
        
        
        
        if action=="Salir":
            return(self.__values[fila_actual,columna_actual],"END")
              
        if action=="up":#moverse hacia arriba, restar 1 al valor de la fila actual
            self.current_state=(fila_actual-1,columna_actual)
        elif action=="down":#moverse hacia arriba, sumar 1 al valor de la fila actual
            self.current_state=(fila_actual+1,columna_actual)
        elif action=="right":#moverse hacia derecha, sumar 1 al valor de la columna actual
            self.current_state=(fila_actual,columna_actual+1)
        else:
            #moverse hacia izquierda, restar 1 al valor de la columna actual
            self.current_state=(fila_actual,columna_actual-1)
        return (self.grid[self.current_state[0],self.current_state[1]],self.current_state)   
    def reset(self):#reiniciar
        self.current_state=(0,0)
        self.reward=0
    def is_terminal(self,estado):
        if (self._is_term[estado[0],estado[1]]==1) | (self._is_term[estado[0],estado[1]]==-1):
            return True
        else:
            return False
        
    #HASTA AQUI EL PUNTO I) DE LA TAREA, QUE CORRESPONDIA A LA DEFINICION DEL AMBIENTE.
    #PARA LA PARTE II) DE LA TAREA, QUE ES CONSTRUIR EL MARKOV DECISSION PROCESS, DEFINIMOS OTRO
    #METODO, QUE LLAMÉ MDP
    def grilla(self):
        colormap = colors.ListedColormap(["red","green","blue"])
        plt.figure(figsize=(5,5))
        plt.imshow(self.grid,cmap=colormap)
        plt.title("Grilla para juego: en verde las casillas sin recompensa, en blanco, los obstaculos, en azul la casilla objetivo, y en rojo las casillas trampa")
        plt.show()
                
    
class Q_LEARNING():
    def __init__(self,env,epsilon,gamma,alpha):
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha
        self.env=env
        #creamos el atributo Q para guardar los valores de las acciones y los estados
        #Inicalizalizamos un diccionario para almacenar los q_valores de las acciones y estados:
        self.Q={}
        filas,columnas=self.env.grid.shape
        acciones=["up","right","left","down","Salir"]
        for i in range(filas):
            for j in range(columnas):
                if np.isnan(self.env.grid[i,j])==False:
                    if self.env.is_terminal((i,j))==True:
                        self.Q[(i,j)]={"Salir":0}
                    else:
                        self.Q[(i,j)]={accion:0 for accion in acciones}
                    
        #inicializamos una lista con todos los posibles estados:
        self.posibles_estados=[]
        for i in range(filas):
            for j in range(columnas):
                if np.isnan(self.env.grid[i,j])==False:
                    self.posibles_estados.append((i,j))
        
        
        
    def choose_action(self,estado):
        #seleccionamos la accion maxima
        accion= max(self.Q[estado], key=self.Q[estado].get)
        #seleccionamos la lista de todas las posibles acciones:
        posibles_acciones=list(self.Q[estado].keys())
        #creamos una lista con las posibles acciones diferentes a la greedy:
        posibles_acciones_diferentes=[]
        for element in posibles_acciones:
            if element!=accion:
                posibles_acciones_diferentes.append(element)
        #ahora escogemos entre seguir la greedy o no:
        
        decision=np.random.choice([0,1],p=[1-self.epsilon,self.epsilon])#0 si sigue la politica greedy, 1 si no
        if decision==0:
            return accion#si salio seguir la greedy, devuelve la accion greedy
        else:
            try:
                return random.choice(posibles_acciones_diferentes)#si salio la opcion de explorar, devuelve una accion 
            except IndexError:
                return accion
                #aleatoria diferente a la greedy
    def Q_max(self,estado):#devuelve el valor del maximo estado al que se puede llegar desde el estado dado:
        accion= max(self.Q[estado], key=self.Q[estado].get)
        return self.Q[estado][accion]
        
                
    def action_function(self,estado1, accion1, recompensa, estado2):
        #implementa el calculo del q_valor de la accion segun el agoritmo de Q_learning
        
        if estado2=="END":
            self.Q[estado1][accion1]=self.Q[estado1][accion1]+self.alpha*(recompensa-
                                                                     self.Q[estado1][accion1])
        else:
            Q_max=self.Q_max(estado2)
            self.Q[estado1][accion1]=self.Q[estado1][accion1]+self.alpha*(recompensa+self.gamma*Q_max-self.Q[estado1][accion1])
    def episodio(self):
        #inicializar un estado S:
        S_1=random.choice(self.posibles_estados)
        
        terminado=False#creamos una variable para saber si el episodio esta terminado
        while terminado==False:
            
            if self.env.is_terminal(S_1)==True:
                #print(S_1)
                recompensa,S_2=self.env.do_action("Salir")#ejecutamos la accion 1 desde el estado 1, y obtenemos la recompensa
                #print("Salir")
                #print(recompensa)
                self.action_function(S_1,"Salir",recompensa,S_2)
                terminado=True
            else:
                           
                #print(S_1)
                accion1=self.choose_action(S_1)#calculamos la accion a tomar para estado S1
                #print(accion1)
                self.env.current_state=S_1
                recompensa,S_2=self.env.do_action(accion1)#ejecutamos la accion 1 desde el estado 1, y obtenemos la recompensa
                #print(recompensa)
                #y quedamos en el estado 2
                
                #finalmente, ejecutamos la funcion action_function:
                self.action_function(S_1,accion1,recompensa,S_2)
                #ahora cambiamos S1 a S2 para el siguiente step

                S_1=S_2
                #verficamos si es terminal
                if accion1=="Salir":
                    terminado=True
    def politica_optima(self,episodios):
        #ejecuta los episodios:
        for i in range(episodios):
            self.episodio()#ejecuta la funcion episodio
            print(f"episodio {i}")
        #ahora calculamos la politica optima
        filas,columnas=self.env.grid.shape#obtenemos el numero de filas y columnas
        decisiones=[]
        self.epsilon=0#ahora solo seguimos politicas greedy
        for i in range(filas):
            lista_fila=[self.choose_action((i,j)) if np.isnan(self.env.grid[i,j])==False else None for j in range(columnas) ]
            decisiones.append(lista_fila)
        return pd.DataFrame(decisiones)
   
        
        
my_board=board()
my_q_learning=Q_LEARNING(my_board,0.5,0.2,0.1)
my_q_learning.politica_optima(100)