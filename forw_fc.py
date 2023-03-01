import numpy as np

'''
  Implementation of Forward Fully Connect activation function
  Input 
        x   : m*n || np.array
        w   : m*n || np.array
        b   : scalar bias value
  Output
        y   : scalar value
'''

def forw_fc(x:np.array,w:np.array,b:np.array):
  xw = 0
  for row_idx in range(len(x)):
    for col_idx in range(len(x[row_idx])):
        xw += x[row_idx,col_idx] * w[row_idx,col_idx]
  y = round(xw, 4) + b
  return y