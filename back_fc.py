import numpy as np
'''
  Implementation of Backward Fully Connect activation function
  Input 
        x   : m*n || np.array
        w   : m*n || np.array
        b   : scalar bias value
        y   : scalar value (output from forward pass)
        dzdy: scalar value dx/dy
  Output
        List: [dzdx,dzdw,dzdb]
        dzdx : m*n matrix of dz/dx_{ij} values
        dzdw : m*n matrix of dz/dW_{ij} values
        dzdb : value of dz/db
'''
def back_fc(x: np.array,w: np.array,b: np.array,y: np.array,dzdy: np.array):
  dzdx = np.zeros_like(x)
  dzdw = np.zeros_like(w)
  dzdb = np.zeros_like(b)
  dzdx = np.round(np.multiply(w, dzdy),4)
  dzdw = np.round(np.multiply(x, dzdy),4)
  dzdb = np.round(1 * dzdy,4)
  return [dzdx,dzdw,dzdb] 