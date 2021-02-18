# NeuralNetworkFrame

### Importante, requiere VisualizeNN, obtener de:
[VisualizeNN](https://github.com/jzliu-100/visualize-neural-network)

Framework para la creación de redes neuronales de una forma sencilla o automática y de tamaño personalizado.

Tiene implementado un sistema de diferenciación automática que usa el descenso del gradiente para la optimización de las redes por medio de la propagación hacia atrás. 

Permite visualizar la estructura de la red, guardarla, entrenarla, obtener los pesos y umbrales, cuenta con las principales funciones de activación e igual es bastante sencillo su uso. 

### Funciones de activación:
*Sigmoide
*Tangente Hiperbolica
*ReLU
*lineal

Funciones:
*La estructura visual de la red puede guardarse como una imagen.
*Se pueden guardar los pesos y umbrales para ser importados más tarde.
*Es posible imprimir los pesos y umbrales como texto o en formato de matriz python para una posterior implementación.
*Imprime los pesos y umbrales en matrices o arreglos con formato C para código arduino o C, C+, C++.
*Es posible ajustar la taza de aprendizaje o "learning rate" e incluso ponerle un "decay" para que conforme vaya acercandose al minimo de la función de la red(el punto óptimo) el lr vaya disminuyendo y así sea más preciso el entrenamiento.

### Requiere:
*Numpy
*MatplotLib
*VisualizeNN
*Palettable
