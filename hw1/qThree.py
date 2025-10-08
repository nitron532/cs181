import numpy as np
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
import os
import heapq
from scipy.signal import convolve2d

pathToBase = os.path.join(os.getcwd(),"upperleftcorner.png") #path to greyscale kavli upperleft corner
baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable
#remove rgba and ensure double values
baseLevel = baseLevel[:,:,0]
baseLevel = baseLevel.astype(np.double)
smoothedKavli = ski.filters.gaussian(baseLevel, sigma = 1) #smooth before taking derivatives
#compute derivatives
Ix = ski.filters.sobel_v(smoothedKavli)
Iy = ski.filters.sobel_h(smoothedKavli)
# print(Ix)
# test = input()
Ixx = np.square(Ix)
Iyy = np.square(Iy)
Ixy = np.multiply(Ix, Iy)
#compute averaged derivatives over the 3 x 3 window
window = np.ones(shape = (3,3)) / 9.0

structureDerivIxx = convolve2d(Ixx, window, mode='same', boundary='symm')
structureDerivIxy = convolve2d(Ixy, window, mode='same', boundary='symm')
structureDerivIyy = convolve2d(Iyy, window, mode='same', boundary='symm')


#stores CRF values in a min heap, flip the sign of the score to accurately reflect top corner scores
CRFHeap = []

#iterate thru all pixels
print("Computing local structure tensors...")
# localStructureTensor = np.array([[structureDerivIxx[413,332], structureDerivIxy[413,332]],\
#                                                [structureDerivIxy[413,332], structureDerivIyy[413,332]]],\
#                                                 dtype = np.double)
# eigenValues, eigenVectors = np.linalg.eig(localStructureTensor)
# print(len(eigenValues), 'size')
# print(eigenValues[0], eigenValues[1])
# eigenValues[0] = 0 if eigenValues[0] < 0 else eigenValues[0]
# eigenValues[1] = 0 if eigenValues[1] < 0 else eigenValues[1]
# k = 0.04
# trace = eigenValues[1] + eigenValues[0]
# determinant = eigenValues[1] * eigenValues[0]
# CRFValue = determinant - (k * (trace**2))
# print(CRFValue)
for i in range(0,512):
    for j in range(0,512):
        #compute local structure tensor A for pixel(i,j)
        localStructureTensor = np.array([[structureDerivIxx[i,j], structureDerivIxy[i,j]],\
                                               [structureDerivIxy[i,j], structureDerivIyy[i,j]]],\
                                                dtype = np.double)
        #compute eigen values
        eigenValues, eigenVectors = np.linalg.eig(localStructureTensor)
        #if negative due to noise, or floating point precision, set to 0
        eigenValues[0] = 0 if eigenValues[0] < 0 else eigenValues[0]
        eigenValues[1] = 0 if eigenValues[1] < 0 else eigenValues[1]
        #compute trace and determinant for CRFValue
        k = 0.04
        trace = eigenValues[1] + eigenValues[0]
        determinant = eigenValues[1] * eigenValues[0]
        CRFValue = determinant - (k * (trace**2))
        #push opposite sign of CRFValue to minheap
        # print(CRFValue, (i,j))
        heapq.heappush(CRFHeap,(-CRFValue, (i,j)))
print("Done!")
#413,332 is a good corner pixel
topCorners = []
heapq.heapify(CRFHeap)
for i in range(0,50):
    topCorners.append(CRFHeap.pop()[1])

kavliWithCorners = np.repeat(baseLevel[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

for coordinatePair in topCorners:
    # print(coordinatePair[0], coordinatePair[1])
    row, col = ski.draw.circle_perimeter(coordinatePair[0], coordinatePair[1], 4, shape = kavliWithCorners.shape)
    kavliWithCorners[row,col,:] = (255,255,0)

dpi = 100
height, width = kavliWithCorners.shape[:2]
figsize = width / dpi, height / dpi
plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(kavliWithCorners, interpolation = 'nearest')
plt.axis("off")
plt.tight_layout(pad = 0)
#label and show the image
plt.show()