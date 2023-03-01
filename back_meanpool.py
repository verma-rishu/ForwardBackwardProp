import numpy as np
'''
  Implementation of Backward Mean Pool activation function
  Input 
        x   : 2m*2n || np.array
        y   : m*n   || np.array (output from forward pass)
        dzdy: m*n matrix of dz/dy_{ij} values
  Output
        dzdx : 2m*2n matrix of dz/dx_{ij} values
'''
def back_meanpool(x:np.array,y:np.array,dzdy:np.array):
  dzdx = np.zeros_like(x)
  stride = 2
  for y_row_idx in range(0,len(y)):
    for y_col_idx in range(0,len(y[y_row_idx])):
      x_row_start_idx = y_row_idx * stride 
      x_col_start_idx = y_col_idx * stride
      curr_val = y[y_row_idx, y_col_idx]
      for x_i in range(0,stride):
        for y_i in range(0,stride):
          dzdx[x_i + x_row_start_idx, y_i + x_col_start_idx] = dzdy[y_row_idx, y_col_idx]/4            

  return dzdx