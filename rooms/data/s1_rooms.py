#definicion de clase qlearning

class qlearning:
    #definición de atributos
    def __init__(self, md, d, episodes, epsi, gam, alp, valGan, valPer, valBloq, inix, iniy):
        #modelo a resolver 
        self.mdp = md

        #values que corresponde a los valores calculados para los estados del modelo.
        self.values = copy.deepcopy(self.mdp)
        #creo una matriz de acciones
        self.action = self.createAction() 
        #politica a evaluar / política inicial
        self.policy = copy.deepcopy(self.mdp)
        #discount que corresponde al factor de decuento a utilizar, 0.9 por defecto.
        if d > 0:
          self.discount = d
        else:
          self.discount = 0.9

        #epsilon
        if epsi > 0:
          self.epsilon = epsi
        else:
          self.epsilon = 0.6

        #gamma
        if gam > 0:
          self.gamma = gam
        else:
          self.gamma = 0.96

        #alpha
        if alp > 0:
          self.alpha = alp
        else:
          self.alpha = 0.81

        #episodes
        self.eps = episodes

        #Q valores
        self.qValue = None
        self.createQvalue()
        self.finalQValue = copy.deepcopy(self.mdp) 

        #Policy noise
        self.bp = {'up':0.3, 'left':0.2, 'right':0.3, 'down':0.2}

        #valores de inicio
        self.valorGanador = valGan
        self.valorPerdida = valPer
        self.valorBloqueo = valBloq

        #reiniciar
        self.stateIni = (inix,iniy)

    def rec_find(self, eps):

        for episode in range(eps):
          if episode < 2:
            print('Episodio ' +  str(episode))
            print('Política')
            self.imprimir(self.policy)
            #imprimo tabla de valores
            print('valores')
            self.imprimir(self.finalQValue)

          self.stateIni
          #recorrer estados - modelo, no conozco el modelo
          estadoFinal = False
          conteo = 0
          while not estadoFinal and conteo < 400:
            #(1) La interacción comienza decidiendo la acción a tomar para el estado actual (la cual esta dada por el agente), 
            action = self.choose_action(state)
            #(2) luego debemos ejecutar la acción, obteniendo el estado de llegada y la recompensa de ejecutar dicha acción, 
            nextState = self.next_action(state, action)
            #(3) luego calculamos la recompensa y si es estado final o salida 
            resultValue = self.get_value(nextState) 
            #(4) por último calculamos el q-valor definido por la función de las acciones.
            #    preguntamos si es estado final
            if resultValue[0] == 'estadoFinal':
              estadoFinal = True
            #(5) se realiza la actualización de valores.
            self.updateQ(state, action, resultValue[1], nextState, estadoFinal)
            conteo = conteo + 1
            state = nextState
          
          #actualizo política y tabla q
          self.finalQvalueF()
          self.finalpol()

          if (episode % 20) == 0:
            print('Episodio ' +  str(episode))
            #imprimo tabla de valores
            print('valores')
            self.imprimir(self.finalQValue)
            print('politica')
            self.imprimir(self.policy)
            #print('qvalue')
            #print(self.values)

          
    def choose_action(self, state):
      #print(' estado para accion ' + str(state))
      action = ''
      if np.random.uniform(0, 1) < self.epsilon:
          action = np.random.choice(self.action[state[0]][state[1]])
      else:
        #maxima acción de tabla q en ese estado
        action = max(self.qValue[state], key = self.qValue[state].get)
      return action

    #next_action 
    def next_action(self, state, action):
        wx = state[1]
        wy = state[0]
        if action == 'up':
          wy = state[0] - 1
        if action == 'down':
          wy = state[0] + 1
        if action == 'left':
          wx = state[1] - 1
        if action == 'right':
          wx = state[1] + 1
        stateNext = (wy,wx)
        return stateNext

    #get_value recibe un estado y retorna el valor correspondiende para dicho estado.
    def get_value(self, state):
        reward = 0
        estado = 'estadonoFinal'
        eval = False
        #print(self.values)
        #print('funcion getval state ' + str(state) + ' valorGanador ' + str(self.valorGanador) + ' valorBloqueo ' + str(self.valorBloqueo) + ' valorPerdida ' + str(self.valorPerdida) + ' valor de matriz ' + str(self.values[state[0]][state[1]]))
        if self.values[state[0]][state[1]] == self.valorGanador and (not eval):
          reward = self.valorGanador
          estado = 'estadoFinal'
          eval = True
        if self.values[state[0]][state[1]] == '*' and (not eval):
          reward = self.valorBloqueo
          eval = True
        if self.values[state[0]][state[1]]==self.valorPerdida and (not eval):
          reward = self.valorPerdida
          estado = 'estadoFinal'
          eval = True
        if not eval:
          reward = self.values[state[0]][state[1]]   
          eval = True
        #print('funcion getval state rward' + str(reward))
        if reward == '':
           reward = 0

        return estado,reward

    def updateQ(self, state, action, reward, nextState, estadoFinal):
      #print('state ' + str(state) + ' act ' + str(action) + ' rew ' + str(reward) + ' nextState ' + str(nextState))
      #print('state qv ' + str(self.qValue[state][action]))
      #print('max ' + str((round(max(self.qValue[nextState].values()),2))))
      #endVal = float(reward)
      #if not estadoFinal:
      endVal = float(reward) + round(float(self.gamma) * float(max(self.qValue[nextState].values())),2)
      ini = float(self.qValue[state][action])
      self.qValue[state][action] = round(ini + float(self.alpha) * (endVal - ini),2)
      #print('ini ' + str(round(ini,2)) + 'fin ' + str(round(endVal,2)))
      #print('sqlval ' + str(self.qValue[state][action]))

    def createAction(self):
      actionsMDP = [[0 for i in range(len(self.mdp[0]))] for j in range(len(self.mdp))]
      #recorrer todos los estados - mdp
      #print('tamaño ' + str(len(self.mdp) - 1) + ' ' + str(len(self.mdp[0]) - 1))
      i=0
      for row in actionsMDP:
        j=0
        for column in row:
          state = (i,j)
          #print('creando lista de estados ' + str(state))
          actions = ()
          if state[0] > 0:
            actions += ('up',)
          if state[0] < len(self.mdp)-1:
            actions += ('down',)
          if state[1] > 0:
            actions += ('left',)
          if state[1] < len(self.mdp[0])-1:
            actions += ('right',)
          actionsMDP[i][j] = actions
          j+=1
        i+=1
      return actionsMDP

    def createQvalue(self):
      self.qValue = {}
      #print('acciones ' + str(self.action))
      i=0
      for row in self.action:
        j=0
        for column in row:
          state = (i,j)
          self.qValue[state]={}
          for eact in self.action[state[0]][state[1]]:
            self.qValue[state][eact]=0.0
          j += 1
        i += 1
      print(self.qValue)      

    def finalQvalueF(self):
      i=0
      for row in self.mdp:
        j=0
        for column in row:
          state = (i,j)
          #print(' max ' + str(self.values[state[0]][state[1]]))
          if self.values[state[0]][state[1]] == '*':
            self.finalQValue[state[0]][state[1]] = '*'
          else:
            self.finalQValue[state[0]][state[1]] = round(float(max(self.qValue[state].values())),2)
          j += 1
        i += 1


    def finalpol(self):
      i=0
      for row in self.policy:
        j=0
        for column in row:
          state = (i,j)
          #print('state ' + str(state) + ' self.qValue[state] ' + str(self.qValue[state]) + ' max ' + str(max(self.qValue[state], key = self.qValue[state].get)))
          if self.values[state[0]][state[1]] == '*':
            self.policy[state[0]][state[1]] = '*'
          else:
            self.policy[state[0]][state[1]] = max(self.qValue[state], key = self.qValue[state].get)
          j += 1
        i += 1

    #imprimir tabla
    def imprimir(self, obj):
      print('                          ')
      i=0
      for row in obj:
        j=0
        linea = ''
        for column in row:
          #print(obj[i][j])
          val = obj[i][j]
          linea += ' ' + str(val) + ' | '
          j+=1
        print(linea)  
        i+=1
      print('                          ')

