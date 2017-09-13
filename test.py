from neural_network import Neural_Network
NN = Neural_Network()
import numpy as np
NN.init_()
X = np.array([[2,3],[4,5],[2,3]])
y = np.array([[0.9],[0.8],[0.4]])
a1, a2 = NN.costFunctionPrime(X,y)

print(a1)
print(a2)
