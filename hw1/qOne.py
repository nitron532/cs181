import numpy as np
from scipy.signal import convolve2d

M = np.array([[1,1,1],[3,3,3],[6,6,6]])
Sx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
Sy = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
N = np.array([[10,10,10],[10,100,100],[10,100,100]])
L = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])

print(convolve2d(Sx, M, mode = "same", boundary = "fill"))
print(convolve2d(Sy, M, mode = 'same', boundary = "fill"))
print()
print(convolve2d(L, N, mode = 'same', boundary = "fill"))