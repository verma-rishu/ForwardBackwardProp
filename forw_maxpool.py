import numpy as np
'''
  Implementation of Forward Max Pool activation function
  Input 
        x   : 2m*2n || np.array
  Output
        y   : m*n   || np.array 
'''
def forw_maxpool(x:np.array):
  height, width = x.shape
  frag = 2
  stride = 2

  newHeight = int((height-frag)/stride)+1
  newWidth = int((width-frag)/stride)+1
  dzdx = np.zeros((newHeight, newWidth))
  
  for row_idx in range(newHeight):
    for col_idx in range(0,newWidth):
      row_start_idx = row_idx * stride 
      col_start_idx = col_idx * stride
      sel_region = x[row_start_idx:row_start_idx+stride, col_start_idx:col_start_idx+stride] 
      dzdx[row_idx, col_idx] = np.amax(sel_region)

  return dzdx