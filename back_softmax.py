import numpy as np
'''
  Implementation of Backward Softmax activation function
  Input 
        x   : m*1 || np.array
        y   : m*1 || np.array (output from forward pass)
        dzdy: m*1 matrix of dz/dy_{ij} values
  Output
        dzdx : m*1 matrix of dz/dx_{ij} values
'''
def back_softmax(x:np.array,y:np.array,dzdy:np.array):
    dzdx = np.zeros_like(x)
    for row_idx in range(len(x)):
        for col_idx in range(len(x)):
            if row_idx == col_idx:
                dzdx[row_idx] += (y[row_idx]* (1-y[row_idx]))*dzdy[col_idx]
            else:
                dzdx[row_idx] -= (y[row_idx]*y[col_idx])*dzdy[col_idx]
    return dzdx