import numpy as np
import skimage as ski
import matplotlib.pyplot as plt
import os
from scipy.signal import convolve2d
from scipy.signal import convolve

def nonMaximumSuppression(points, shape, radius=5):
    suppressed = np.zeros(shape, dtype=bool)
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


def harrisCorner(fileName, sigma, top,radius): #color png that must be in the same directory as calling file
    # pathToBase = os.path.join(os.getcwd(),fileName) #path to greyscale kavli upperleft corner
    # baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable
    #remove rgba and ensure double values
    baseLevel = fileName #if you pass in an ndarray
    # baseLevel = baseLevel[:,:,1] #enable this for kavli
    baseLevel = baseLevel.astype(np.float32)
    # baseLevel = ski.color.rgb2gray(baseLevel)
    if sigma != -1:
        baseLevel = ski.filters.gaussian(baseLevel, sigma)
    #compute derivatives

    Ix = ski.filters.sobel_v(baseLevel)
    Iy = ski.filters.sobel_h(baseLevel)

    Ixx = np.square(Ix)
    Iyy = np.square(Iy)
    Ixy = np.multiply(Ix, Iy)

    #compute averaged derivatives over the 3 x 3 window by convoluting 
    window = np.ones(shape = (3,3)) / 9.0

    structureDerivIxx = convolve2d(Ixx, window, mode='same', boundary='fill')
    structureDerivIxy = convolve2d(Ixy, window, mode='same', boundary='fill')
    structureDerivIyy = convolve2d(Iyy, window, mode='same', boundary='fill')

    #stores CRF values but not in correct order yet
    topCorners = []

    #iterate thru all pixels and compute structure tensors
    numRows, numCols = baseLevel.shape
    for i in range(0,numRows):
        for j in range(0,numCols):
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

    topCorners.sort(reverse = True) #sort reverse to put largest CRF values (corners) at the front (sort normal order for edges)
    CRFTopCornerCoords = nonMaximumSuppression(topCorners, baseLevel.shape) #choose best features in areas
    CRFTopCornerCoords = CRFTopCornerCoords[:top] #top (top) corners or edges depending on sort order

    display_img = np.stack([baseLevel]*3, axis=-1)     # shape (h,w,3)
    display_img = (display_img*255).astype(np.uint8) # convert 0-1 float to uint8
    #add the *255 for black square problem
    # plt.imshow(display_img, cmap = "gray", interpolation= "nearest")
    # plt.show()

# Draw red circles at corners
    for value, (r, c) in CRFTopCornerCoords:
        rr, cc = ski.draw.circle_perimeter(int(r), int(c), radius, shape=baseLevel.shape)
        display_img[rr, cc] = [255, 0, 0]  # red (BGR if OpenCV, RGB here)

    plt.imshow(display_img, cmap = "gray", interpolation = 'nearest')
    plt.show()
    # for value, coord in CRFTopCornerCoords:
    #     print(coord[1],",", coord[0])

# b = ski.io.imread("upperleftcorner.png")
# harrisCorner(b, 1, 50,4)