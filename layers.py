import numpy as np
from abc import abstractmethod
from activation import Sigmoid

class Layers:
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def backward(self):
        pass

class Dense(Layers):
    def __init__(self, num_units, initial_weights , initial_biases):
        # super().__init__()
        self.units = num_units
        self.W =initial_weights
        self.b =initial_biases

    def forward(self, x):
        self.inputs = x
        return np.matmul(x,self.W.T)+self.b

    def backward(self, global_grad):
        #with respect to x
        # return np.dot(global_grad , W.T) 
        pass

    def gradient(self):
        return self.W.T

class Activations(Layers):
    activations = {
        "Sigmoid":Sigmoid
    }
    def __init__(self, name):
        self._func = self.activations[name]()

    def forward(self, x):
        self.inputs = x
        return self._func.forward(x)

    def backward(self, global_grad):
        return global_grad*self._func.gradient(self.inputs)