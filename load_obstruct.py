import random
import pandas as pd
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from numba import jit
import matplotlib.colors as colors

def obstruct_arrays(obstruct):
  rows,cols = obstruct.shape
  obstruct_u = np.ones((rows,cols+1))
  for row in range(0,rows):
    obstruct_u[row][0] = 0
    obstruct_u[row][cols] = 0
  for row in range(0,rows):
    for col in range(0,cols):
      if col == 0 and obstruct[row][col] == 99:
        obstruct_u[row][col] = 'nan'
      elif col == cols-1 and obstruct[row][col] == 99:
        obstruct_u[row][col+1] = 'nan'
      else:
        if obstruct[row][col] == 99 and obstruct[row][col-1] == 99:
          obstruct_u[row][col] = 'nan'
        if obstruct[row][col] == 99 and obstruct[row][col+1] == 99:
          obstruct_u[row][col+1] = 'nan'
        if obstruct[row][col] == 99 and obstruct[row][col-1] == 0:
          obstruct_u[row][col] = 0
        if obstruct[row][col] == 0 and obstruct[row][col-1] == 99:
          obstruct_u[row][col] = 0

  obstruct_v = np.ones((rows+1,cols))
  for col in range(0,cols):
    obstruct_v[0][col] = 0
    obstruct_v[rows][col] = 0
  for row in range(0,rows):
    for col in range(0,cols):
      if row == 0 and obstruct[row][col] == 99:
        obstruct_v[row][col] = 'nan'
      elif row == rows-1 and obstruct[row][col] == 99:
        obstruct_v[row+1][col] = 'nan'
      else:
        if obstruct[row][col] == 99 and obstruct[row-1][col] == 99:
          obstruct_v[row][col] = 'nan'
        if obstruct[row][col] == 99 and obstruct[row+1][col] == 99:
          obstruct_v[row+1][col] = 'nan'
        if obstruct[row][col] == 99 and obstruct[row-1][col] == 0:
          obstruct_v[row][col] = 0
        if obstruct[row][col] == 0 and obstruct[row-1][col] == 99:
          obstruct_v[row][col] = 0

  obstruct_copy = np.zeros((rows,cols))
  for row in range(0,rows):
    for col in range(0,cols):
      if obstruct[row][col]==0:
        obstruct_copy[row][col] = 'nan'
      else:
        obstruct_copy[row][col] = 99
  return obstruct_u, obstruct_v, obstruct_copy
