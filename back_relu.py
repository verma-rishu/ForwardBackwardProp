import numpy as np
'''
  Implementation of Backward ReLU activation function
  Input 
        x   : m*n || np.array
        y   : m*n || np.array (output from forward pass)
        dzdy: m*n matrix of dz/dy_{ij} values
  Output
        dzdx : m*n matrix of dz/dx_{ij} values
'''
def back_relu(x:np.array,y:np.array,dzdy:np.array):
  dzdx = np.zeros_like(x)
  for row_idx in range(0,len(x)):
    for col_idx in range(0,len(x[row_idx])):
      #dydx = 0
      if(x[row_idx, col_idx] > 0):
        #dydx = 1
        dzdx[row_idx,col_idx] = dzdy[row_idx,col_idx]#*dydx  
  return dzdx