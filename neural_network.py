import numpy as np
class Neural_Network(object):
    def init_(self):
        #Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerUnitSize = 3

        #Weights Parameters
        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerUnitSize)
        self.W2 = np.random.randn(self.hiddenLayerUnitSize,self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self,z):
        #Applied sigmoid activation function
        return 1/(1 + np.exp(-z))

