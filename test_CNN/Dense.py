import NN_basics.layer as Layer
import numpy as np 

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1) # column vector (y = Mx+b)
        
    def forward(self, input):
        self.input = input 
        return np.dot(self.weights, self.input)+self.bias
    
    # dE/dW = dE/dY * X^t
    # dE/dB = dE/dY
    # dE/dX = dE/dY * W^t
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input) + self.bias 
        self.weights -= learning_rate*weights_gradient
        self.bias -= learning_rate*output_gradient
        return np.dot(self.weights.T, output_gradient)
    
    
         
    
    