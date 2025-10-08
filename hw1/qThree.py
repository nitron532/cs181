import numpy as np
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
import os
import heapq
import cv2
from scipy.signal import convolve2d
from scipy import ndimage

def nonMaximumSuppression(points, radius=5):
    suppressed = np.zeros((512,512), dtype=bool)
    kept = []

    for response, coord in points:
        y = coord[1]
        x = coord[0]
        if suppressed[y, x]:
            continue  # already suppressed by a stronger neighbor

        kept.append((response, (x, y)))

        # Suppress neighbors within radius
        yMin = max(0, y - radius)
        yMax = min(512, y + radius + 1)
        xMin = max(0, x - radius)
        xMax = min(512, x + radius + 1)

        suppressed[yMin:yMax, xMin:xMax] = True

    return kept

pathToBase = os.path.join(os.getcwd(),"upperleftcorner.png") #path to greyscale kavli upperleft corner
baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable
#remove rgba and ensure double values
baseLevel = baseLevel[:,:,0]
baseLevel = baseLevel.astype(np.float32)
smoothedKavli = ski.filters.gaussian(baseLevel, sigma = 2) #smooth before taking derivatives
#compute derivatives
Ix = ski.filters.sobel_v(smoothedKavli)
Iy = ski.filters.sobel_h(smoothedKavli)

Ixx = np.square(Ix)
Iyy = np.square(Iy)
Ixy = np.multiply(Ix, Iy)

#compute averaged derivatives over the 3 x 3 window by convoluting 
window = np.ones(shape = (3,3)) / 9.0

structureDerivIxx = convolve2d(Ixx, window, mode='same', boundary='symm')
structureDerivIxy = convolve2d(Ixy, window, mode='same', boundary='symm')
structureDerivIyy = convolve2d(Iyy, window, mode='same', boundary='symm')

#stores CRF values but not in correct order yet
topCorners = []

#iterate thru all pixels and compute structure tensors
print("Computing local structure tensors...")

for i in range(0,512):
    for j in range(0,512):
        #compute local structure tensor A for pixel(i,j)
        localStructureTensor = np.array([[structureDerivIxx[i,j], structureDerivIxy[i,j]],\
                                               [structureDerivIxy[i,j], structureDerivIyy[i,j]]],\
                                                dtype = np.float32)
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
        topCorners.append((CRFValue, (i,j)))
print("Done!")

topCorners.sort(reverse = True) #sort reverse to put largest CRF values (corners) at the front (sort normal order for edges)
CRFTopCornerCoords = nonMaximumSuppression(topCorners) #choose best features in areas
CRFTopCornerCoords = CRFTopCornerCoords[:50] #top 50 corners or edges depending on sort order

#add extra dimension for color on circles
kavliWithCorners = np.repeat(baseLevel[:, :, np.newaxis], 3, axis=2).astype(np.uint8)

#draw circles at best corners/edges
for value, coordinatePair in CRFTopCornerCoords:
    row, col = ski.draw.circle_perimeter(coordinatePair[0], coordinatePair[1], 4, shape = kavliWithCorners.shape)
    kavliWithCorners[row,col,:] = (255,255,0)

#setup for image display through matplotlib
dpi = 100
height, width = kavliWithCorners.shape[:2]
figsize = width / dpi, height / dpi
plt.figure(figsize=figsize, dpi=dpi)
plt.imshow(kavliWithCorners, interpolation = 'nearest')
plt.axis("off")
plt.tight_layout(pad = 0)
#label and show the image
plt.show()
