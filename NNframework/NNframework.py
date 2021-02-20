# El código es autoría de Raúl Góngora Vázquez.
"""
NeuralNetworkFramework.v_3 es una clase para crear redes neuronales de una forma más sencilla;
se basa en el descenso del gradiente y el backpropagation para el entrenamiento de la red,
usa las funciones: "error cuadratico medio" y "error absoluto".
\nCuenta con varias funciónes de activación y también se pueden guardar las redes
generadas para predicciones posteriores. 
\nPara escalado de datos llamar a la función map(). Ver mas en help()
\nLas matrices de pesos y umbrales pueden ser exportadas al lenguaje C.
\nLos datos de entrenamiento deben de ser arreglos de dos dimensiones = [[]].
"""
import numpy as np
import matplotlib.pyplot as plt
import VisualizeNN as VisNN
import os

class NeuralNetwork:
    """
    Para crear una red llamar a la función createNetwork().

    Parametros
    ----------
    action = identificador:
        'create' : crear red.
        'load' : cargar red.
        'help' : pedir ayuda.

    Si no recibe nada se puede llamar a las funciones posteriormente.
        Para cargar una red, llamar a la función loadNetwork().
        Para guardar la red llamar a la función saveNetwork().
        Llamar a help() para más información.
    """
    def __init__(self, action=None):
        if action == 'create':
            self.createNetwork()
        
        elif action == 'load':
            PATH = input('introducir ruta')
            self.loadNetwork(PATH)

        elif action == 'help':
            self.help()
        
        else:
            pass

    def init(self):
        # VECTORES Y MATRICES DE LA RED
        self.networkWeights = []  # PESOS
        self.networkBias = []  # UMBRALES
        self.networkStructure = []  # MEDIDAS DE CADA CAPA DE NEURONAS
        self.functions = []  # NOMBRE DE LA FUNCION POR CAPA
        # MATRICES DONDE SE GUARDAN LAS SALIDAS DE LAS MATRICES ACTIVADAS
        self.activeNeurons = []
        self.networkDeltas = []  # DELTAS PARA EL AJUSTE DE LOS PESOS Y UMBRALES
        self.errorF = 'mse'
        print('Objeto creado.')

    def createNetwork(self, networkStructure=None, functions=None):
        """
        Crea una red automaticamente dejando en None, o introduce los datos para crearla manualmente:
        NeuralNetwork(networkStructure, functions).
        Parametros
        ----------
            networkStructure = vector con el numero de neuronas en cada capa + el numero de entradas.
            Ejemplo: [2, 3, 4, 5], donde el 2 representa el tamaño del vector de entrada, el 5 el tamaño del vector de
            salida, en medio van las capas ocultas y hay que indicar el número de neuronas en cada una.
            
            functions = vector de funciones de activación.
            Ejemplo: [f, f, f] la capa de entrada no tiene función.
            Funciones: "sigmoid", "relu", "tanh", "linear".

            El tamaño del vector debe ser igual al numero de capas.
        """
        self.init()
        # VARIABLES DE LOS DATOS DE LA RED
        if networkStructure:
            self.inputSize = networkStructure[0]
            self.outputSize = networkStructure[len(networkStructure) - 1]
            self.HidenLayers = len(networkStructure) - 2
        else:
            self.inputSize = int(
                input('Introduce el tamaño del vector de ENTRADA'))
            self.outputSize = int(
                input('Introduce el tamaño del vector de SALIDA'))
            self.HidenLayers = int(
                input('Introduce el numero de CAPAS OCULTAS de la red'))

        self.networkLen = self.HidenLayers + 2

        # SE CREA UN VECTOR CON EL NUMERO DE NEURONAS DE CADA CAPA
        if networkStructure:
            self.networkStructure = networkStructure

        else:
            self.networkStructure.append(self.inputSize)
            for i in range(self.HidenLayers):
                self.networkStructure.append(
                    int(input('Numero de neuronas en la capa {}'.format(i+1))))
            self.networkStructure.append(self.outputSize)

            # SELECCIONAMOS LA FUNCIÓN DE ACTIVACIÓN
        if functions:
            for i in functions:
                if(i == "sigmoid"):
                    self.functions.append(i)
                elif(i == "relu"):
                    self.functions.append(i)
                elif(i == "tanh"):
                    self.functions.append(i)
                elif(i == "linear"):
                    self.functions.append(i)
                else:
                    print('EL VECTOR DE FUNCIONES ES INCORRECTO')
                    break

        else:
            for i in range(self.HidenLayers + 1):
                while(True):
                    f = input(
                        'Función de activacion("sigmoid", "relu", "tanh", "linear"(enter)) de la capa: {}'.format(i+1))
                    if(f == "sigmoid"):
                        print('{} agregada en la capa {}'.format(f, i+1))
                        break
                    elif(f == "relu"):
                        print('{} agregada en la capa {}'.format(f, i+1))
                        break
                    elif(f == "tanh"):
                        print('{} agregada en la capa {}'.format(f, i+1))
                        break
                    else:
                        if(f == ''):
                            print('linear agregada en la capa {}'.format(i+1))
                            break
                        else:
                            print('INTRODUCE UNA FUNCION VALIDA')
                self.functions.append(f)

        # SE CREAN PESOS ALEATORIOS CON RANGO(0, 1)
        for i in range(self.HidenLayers + 1):
            self.networkWeights.append(np.random.rand(self.networkStructure[i], self.networkStructure[i+1]))

        # SE CREAN LOS UMBRALES INICIALIZADOS EN 0
        for i in range(self.HidenLayers + 1):
            self.networkBias.append(np.zeros((1, self.networkStructure[i+1]), dtype=float))
        print('Red creada')

    def destroy(self):
        """
        Destruye la red, no borra el objeto. Permite rehacer una red bajo el mismo nombre.
        """
        self.networkWeights = []  # PESOS
        self.networkBias = []  # UMBRALES
        self.networkStructure = []  # MEDIDAS DE CADA CAPA DE NEURONAS
        self.functions = []  # NOMBRE DE LA FUNCION POR CAPA
        # MATRICES DONDE SE GUARDAN LAS SALIDAS DE LAS MATRICES ACTIVADAS
        self.activeNeurons = []
        self.networkDeltas = []  # DELTAS PARA EL AJUSTE DE LOS PESOS Y UMBRALES
        print('La red neuronal fue destruida.')

    def loadNetwork(self, PATH):
        """
        Carga una red guardada:
        Parametros
        ----------
            PATH = ruta donde esta guardada la red, debe ser de la forma: carpeta/
            Ejemplo: carpeta/carpeta/
        """
        self.init()
        self.networkStructure = np.load(PATH + '/networkStructure.npz')
        self.networkStructure = self.networkStructure.f.arr_0
        self.inputSize = self.networkStructure[0]
        self.outputSize = self.networkStructure[(len(self.networkStructure) - 1)]
        self.HidenLayers = len(self.networkStructure) - 2
        self.networkLen = self.HidenLayers + 2
        self.functions = np.load(PATH + '/functions.npz')
        self.functions = self.functions.f.arr_0

        for i in range(self.networkLen - 1):
            z = np.load(PATH + '/W{}.npz'.format(i))
            z = z.f.arr_0
            self.networkWeights.append(z)

        for i in range(self.networkLen - 1):
            z = np.load(PATH + '/b{}.npz'.format(i))
            z = z.f.arr_0
            self.networkBias.append(z)
        print('Red cargada')
    
    def print(self, **kwargs):  # IMPRIME AMBOS TENSORES
        """
        Imprime los pesos y los umbrales:
        Parametros:
        -----------
        T = identificador:
            None = pesos y umbrales.
            'w' = pesos.
            'b' = umbrales.
        
        l = identificador:
            False = vista como matrices
            True = formato para copiar y pegar en código python.
        """
        T = kwargs.get('T', None)
        l = kwargs.get('l', False)

        if (T == None or T == 'w'):
            print('# -------------- W --------------\n')
            for i in range(self.networkLen - 1):
                if (l==True):
                    print('W{}'.format(i+1) +'['+ str(self.networkWeights[i].shape[0])+']['+ str(self.networkWeights[i].shape[1])+'] = '+ str(self.networkWeights[i].tolist()))
                else:
                    print('W{}'.format(i+1))
                    print(self.networkWeights[i])
        if (T == None or T == 'b'):
            print('\n# -------------- b --------------\n')
            for i in range(self.networkLen - 1):
                if (l==True):
                    print('b{}'.format(i+1) +'['+ str(self.networkBias[i].shape[0])+']['+ str(self.networkBias[i].shape[1])+'] = '+ str(self.networkBias[i].tolist()))
                else:
                    print('b{}'.format(i+1))
                    print(self.networkBias[i])

    def visualNN(self):
        """
        Permite visualizar la estructura de la red.
        """
        network_structure = np.hstack(self.networkStructure)
        network = VisNN.DrawNN(network_structure)
        network.draw()

    # FUNCIONES DE ACTIVACIÓN
    def linear(self, s, deriv=False):  # FUNCIÓN LINEAL
        if (deriv == True):
            return 1
        return s

    def sigmoid(self, s, deriv=False):  # FUNCIÓN SIGMOIDE
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

    def tanh(self, s, deriv=False):  # FUNCIÓN TANGENTE HIPERBOLICA
        if (deriv == True):
            return 1 - s**2
        return (2/(1 + np.exp(-2*s))) - 1

    def relu(self, s, deriv=False):  # FUNCIÓN RECTIFIER LINEAR UNIT
        if (deriv == True):
            s[s <= 0] = 0
            s[s > 0] = 1
            return s
        return np.maximum(s, 0, s)

    def function(self, layer, X, deriv=False): # SELECCIONA LA FUNCIÓN DE LA CAPA DE ACUERDO A SU INDICE
        if(self.functions[layer] == 'sigmoid'):
            a = self.sigmoid(X, deriv)
        elif(self.functions[layer] == 'relu'):
            a = self.relu(X, deriv)
        elif(self.functions[layer] == 'tanh'):
            a = self.tanh(X, deriv)
        elif(self.functions[layer] == 'linear'):
            a = self.linear(X, deriv)
        return a

    def predict(self, X):
        """
        Predice la salida a partir del vector de entrada.
        """
        if self.activeNeurons:
            self.activeNeurons[0] = np.dot(np.array(X), self.networkWeights[0]) + self.networkBias[0]
            self.activeNeurons[0] = self.function(0, self.activeNeurons[0])
            for i in range(self.HidenLayers):
                self.activeNeurons[i+1] = np.dot(self.activeNeurons[i], self.networkWeights[i+1]) + self.networkBias[i+1]
                self.activeNeurons[i+1] = self.function(i+1, self.activeNeurons[i+1])
        else:
            self.activeNeurons.append(np.dot(X, self.networkWeights[0]) + self.networkBias[0])
            self.activeNeurons[0] = self.function(0, self.activeNeurons[0])

            for i in range(self.HidenLayers):
                self.activeNeurons.append(np.dot(self.activeNeurons[i], self.networkWeights[i+1]) + self.networkBias[i+1])
                self.activeNeurons[i+1] = self.function(i+1, self.activeNeurons[i+1])

        output = self.activeNeurons[self.HidenLayers]
        return output

    def mean(self, B):  # CALCULA UN PROMEDIO DE LOS DELTAS PARA AJUSTAR LOS UMBRALES
        b, c = np.matrix(B).shape
        a = np.zeros((1, c))
        for j in range(c):
            for i in range(b):
                a[0][j] += B[i][j]
        return a

    def e(self, Y, output, deriv=True):
        Y = np.array(Y)
        output = np.array(output)
        if self.errorF == 'mae':
            if deriv == False:
                e = (1/self.outputSize)*abs(Y - output)
                return np.mean(e)
            e = []
            for k in range(len(Y)):
                _ = []
                for i, j in Y[k], output[k]:
                    _.append(abs(i - j))
                e.append(_)
            e = np.array(e)
            return e

        else:
            if deriv == False:
                e = (1/self.outputSize)*((Y - output)**2)
                return np.mean(e)
            return (2/self.outputSize)*(-(Y - output))
   
    def backPropagation(self, X, Y, Lr, output):  # SE AJUSTAN LOS PESOS Y BIAS CON EL DESCENSO DEL GRADIENTE
        # ERROR PROPAGADO ATRAVEZ DE LA RED
        self.err = self.e(Y, output)
        # CREAMOS LOS DELTAS
        j = self.HidenLayers
        if self.networkDeltas:
            self.networkDeltas[0] = self.err  # error in output
            self.networkDeltas[0] *= self.function(j, output, deriv=True)
            for i in range(1, self.networkLen-1):
                self.networkDeltas[i] = self.networkDeltas[i-1].dot(self.networkWeights[j].T)
                self.networkDeltas[i] *= self.function(j-1, self.activeNeurons[j-1], deriv=True)
                j -= 1
        else:
            self.networkDeltas.append(self.err)  # error in output
            self.networkDeltas[0] *= self.function(j, output, deriv=True)
            for i in range(1, self.networkLen-1):
                self.networkDeltas.append(self.networkDeltas[i-1].dot(self.networkWeights[j].T))
                self.networkDeltas[i] *= self.function(j-1, self.activeNeurons[j-1], deriv=True)
                j -= 1

        # DESCENSO DEL GRADIENTE

        # AJUSTE DE PESOS
        # SE AJUSTA LA PRIMERA MATRIZ DE PESOS
        self.networkWeights[0] -= Lr*(X.T.dot(self.networkDeltas[self.HidenLayers]))

        j = self.HidenLayers - 1
        for i in range(self.HidenLayers):
            self.networkWeights[i+1] -= Lr*(self.activeNeurons[i].T.dot(self.networkDeltas[j]))
            j -= 1

        # AJUSTE DE UMBRALES
        # SE AJUSTA EL PRIMER VECTOR DE UMBRALES
        self.networkBias[self.HidenLayers] -= Lr*self.mean(self.networkDeltas[0])

        j = self.HidenLayers
        for i in range(j):
            self.networkBias[i] -= Lr*self.mean(self.networkDeltas[j])
            j -= 1
            
    def train(self, X, Y, lr, **kwargs):
        """
        Entrena a la red con backpropagation y el descenso del gradiente
        Parametros
        ----------
            X = datos de entrenamientos.
            Y = etiquetas.
            lr = lr de inicio.
        
         kwargs : 
            decay = lr objetivo.
            epochs = epocas a entrenar.
            graph = False(No se imprime grafica).   
            period = Divide el entrenamiento en "period" partes, default = 1.
        """
        decay = kwargs.get('decay', lr)
        epochs = kwargs.get('epochs', len(X)*100)
        period = kwargs.get('period', 1)
        graph = kwargs.get('graph', True)

        X = np.array(X)
        Y = np.array(Y)
        self.error = []
        if epochs < 1000:
            self.mainTrain(X, Y, lr, decay, epochs, graph, self.error)

        else:
            epochs = round(epochs/period)
            for _ in range(period):
                self.mainTrain(X, Y, lr, decay, epochs, graph, self.error)

        print('[accuracy = {}]'.format((1 - self.error[len(self.error)-1])))

        if graph == True:
            self.plot(self.error)

    def mainTrain(self, X, Y, lr, decay, epochs, graph, error):
        for i in range(epochs):  # ENTRENA A LA RED "epochs" VECES
            output = self.predict(X)
            e = self.e(Y, output, deriv=False)
            if (i % (epochs/20) == 0):
                print(self.errorF + ": " + str(e))
            error.append(e)
            if (i % 10 == 0):
                if lr > decay:
                    lr -= decay
            self.backPropagation(X, Y, lr, output)

    def plot(self, error):
        plt.plot(range(len(error)), error, color='red')
        plt.ylim([0, 0.4])
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.tight_layout()
        plt.show()

    def saveNetwork(self, PATH):
        """
        Guarda la red:
        Parametros
        ----------
            PATH = ruta donde se guardará la red debe ser de la forma: carpeta/
            Ejemplo: carpeta/carpeta/
        """
        if os.path.isdir(PATH):
            pass
        else:
            os.mkdir(PATH)

        np.savez(PATH + 'networkStructure.npz', self.networkStructure)
        np.savez(PATH + 'functions.npz', self.functions)

        for i in range(len(self.networkWeights)):
            np.savez(PATH + 'W{}.npz'.format(i), self.networkWeights[i])

        for i in range(len(self.networkBias)):
            np.savez(PATH + 'b{}.npz'.format(i), self.networkBias[i])
        print('Red neuronal guardada con exito')

    def to_str(self, name, W, lib=False):
        if (lib == True):
            s = str(W.tolist()).replace('[', '{').replace(']', '}')
            return 'mtx_type '+name+'['+str(W.shape[0])+']['+str(W.shape[1])+'] = ' + s + ';'

        s = str(W.tolist()).replace('[', '{').replace(']', '}')
        return 'float '+name+'['+str(W.shape[0])+']['+str(W.shape[1])+'] = ' + s + ';'

    def printForArduino(self, lib=False):
        """
        Imprime los pesos y los umbrales en formato de arrays para el IDE Arduino
        Parametros
        ----------
            lib = True
            imprime los arreglos en formato para la libreria MatrixMath
        """
        print('//Weights:\n')
        for i in range(len(self.networkWeights)):
            print(self.to_str('W{}'.format(i), np.matrix(
                self.networkWeights[i]), lib))

        print('\n//Bias:\n')
        for i in range(len(self.networkBias)):
            print(self.to_str('b{}'.format(i),
                              np.matrix(self.networkBias[i]), lib))
        print('\n//Function list:\n')
        print('//{}'.format(self.functions))

    def help(self):
        print("""
        La clase NeuralNetwork tiene las siguientes funciónes:\n
        \tcreateNetwork(): Crea una red.
        \tloadNetwork(): Carga una red.
        \tvisualNN(): Imprime una imagen de la red.
        \tprint(): Imprime pesos y umbrales, juntos o por separado
        \ttrain(): Entrena a la red.
        \tpredict(): Predice las salida a partir de un vector.
        \tsaveNetwork(): Guarda la red.
        \tdestroy(): Permite rehacer una red, borrando su contenido.
        \tPara escalar los valores de los vectores llamar a la función: map(x, in_min, in_max, out_min, out_max)
        \tEsta función no pertenece al objeto NeuralNetwork.
        """)

def map(x, in_min, in_max, out_min, out_max):
    """
    Convierte valores de un rango a otro
    Parametros
    ----------
        x = variable o arreglo
        in_min = valor mínimo de entrada
        in_max = valor máximo de entrada
        out_min = nuevo valor mínimo
        out_max = nuevo valor máximo
    """
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def __version__():
    print('NeuralNetworkFramework v3.1')
