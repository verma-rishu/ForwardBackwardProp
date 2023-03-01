import numpy as np
'''
  Implementation of Forward Softmax activation function
  Input 
        x   : m*1 || np.array
  Output
        y   : m*1 || np.array
'''
def forw_softmax(x:np.array):
  normalizedX = x-np.max(x)
  expo = np.exp(normalizedX)
  return np.round(expo/np.sum(expo),4)