import numpy as np
'''
  Implementation of Forward ReLU (Rectificed Linear Unit) activation function
  Input 
        x   : m*n || np.array
  Output
        y   : m*n || np.array 
'''
def forw_relu(x: np.array):
  return np.maximum(0,x)