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

    def costFunction(self, X, y):
        #Copmpute Cost for given X and y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*np.sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Copmpute derivative with repsect to W1 and W2 for a given X and y.
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def sigmoid(self,z):
        #Applied sigmoid activation function.
        return 1/(1 + np.exp(-z))

    def sigmoidPrime(self,z):
        #Derivative of sigmoid activation function with respect to z.
        return np.exp(-z)/(1 + np.exp(-z))**2
