import numpy as np
from abc import abstractmethod

class Activations:

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def gradient(self):
        pass

class Sigmoid(Activations):
    def forward(self,x):
        # return super().forward()
        return 1/(1+np.exp(-x))

    def gradient(self,x):
        # return super().gradient()
        return self.forward(x)*(1-self.forward(x))

class Swish(Activations):
    def forward(self, x):
        return x*Sigmoid.forward(self,x)
    
    def gradient(self,x):
        return x*Sigmoid.gradient(self , x)+ Sigmoid.forward(self , x)
