import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import cv2 as cv
import os
from scipy import ndimage

def downSampleByHalf(image):
    return image[::2,::2]

def showMPL(image):
    plt.imshow(image, cmap = "gray", interpolation = "nearest")
    plt.show()

def showMPLDots(rgb,x,y):
    plt.imshow(rgb)
    plt.autoscale(False)
    plt.plot(x,y, 'ro')
    plt.show()

pathToBase = os.path.join(os.getcwd(),"upperleftcorner.png") #path to greyscale kavli upperleft corner

baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable

original = baseLevel

baseLevel = baseLevel[:, :, 0] #remove any rgba value
baseLevel = (baseLevel - baseLevel.min()) / (baseLevel.max()-baseLevel.min() + 1e-8)
#1
DOGScaleSpace = []

gaussianOctaves = []

sigmaLevel = 0

for i in range(0,3):
    gaussians = []
    for j in range (0,5):
        sigma = 1.6 * (1.26**sigmaLevel)
        sigmaLevel+=1
        g = ndimage.gaussian_filter(baseLevel, sigma)
        gaussians.append(g)
    gaussianOctaves.append(gaussians)
    DOGoctave = []
    for k in range(1, len(gaussians)):
        D = gaussians[k] - gaussians[k-1]
        DOGoctave.append(D)
    DOGScaleSpace.append(DOGoctave)
    baseLevel = downSampleByHalf(gaussians[-1])

#2
extrema = [] #y,x
#might wanna collect octave and scale indexes
octaveIndex = 0 #which octave it is in (both DOG and Gaussian octave amounts should be the same)
for octave in DOGScaleSpace:
    for i in range(1,len(octave)-1): # i = which image in the octave
        prev, curr, next_ = octave[i-1], octave[i], octave[i+1]
        scaleSigmaLevel = 1.6 * (1.26**(octaveIndex * i ))
        #used gaussian for derivatives since it reduced keypoint detection in the sky
        Dx = ndimage.gaussian_filter(curr,scaleSigmaLevel, order = (0,1))
        Dy = ndimage.gaussian_filter(curr,scaleSigmaLevel, order = (1,0))
        for y in range(1, curr.shape[0]-1):
            for x in range(1, curr.shape[1]-1):
                Dxx = np.square(Dx[y][x])
                Dyy = np.square(Dy[y][x])
                Dxy = np.multiply(Dx[y][x],Dy[y][x])
                # Dxx = curr[y, x+1] + curr[y, x-1] - 2*curr[y, x]
                # Dyy = curr[y+1, x] + curr[y-1, x] - 2*curr[y, x]
                # Dxy = (curr[y+1,x+1] - curr[y+1,x-1] - curr[y-1,x+1] + curr[y-1,x-1])/4
                #3a and b
                TrH = Dxx + Dyy
                DetH = Dxx*Dyy - Dxy**2
                hessianRatio = TrH**2 / (DetH + 1e12)
                if DetH <= 0 or hessianRatio > ((11**2)/10):
                    continue
                currentValue = curr[y,x]
                cube = np.stack([prev[y-1:y+2, x-1:x+2],
                curr[y-1:y+2, x-1:x+2],
                next_[y-1:y+2, x-1:x+2]])
                flat = cube.flatten()
                center_index = 13  # index of curr[y,x] in flattened 3x3x3 cube
                flat = np.delete(flat, center_index)
                if currentValue >= np.max(flat) - 1e-6 and currentValue >= 0.03:
                    extrema.append((octaveIndex, i, y, x, currentValue))
                elif currentValue <= np.min(flat) + 1e-6 and np.abs(currentValue) >= 0.03:
                    extrema.append((octaveIndex, i, y, x, currentValue))
    octaveIndex +=1


#4
#a

plt.imshow(original)
#if keeping track of which gaussian to use gets hard, could keep track of sigma levels per DOG and Gaussian to match them (some sort of class)
for fiveT in extrema:
    scaleSigmaLevel = 1.6 * (1.26**(fiveT[0] * fiveT[1]))
    windowRadius = int(round(3 * scaleSigmaLevel))
    #expand for boundary pixels:
    patch = gaussianOctaves[fiveT[0]][fiveT[1]][fiveT[2] - windowRadius -1 : fiveT[2] + windowRadius + 2,
              fiveT[3] - windowRadius-1 : fiveT[3] + windowRadius + 2]
    ycoords, xcoords = np.mgrid[0: windowRadius*2+3, 0: windowRadius*2+3]
    #dont expand:
    # patch = gaussianOctaves[fiveT[0]][fiveT[1]][fiveT[2] - windowRadius : fiveT[2] + windowRadius + 1,
    #           fiveT[3] - windowRadius : fiveT[3] + windowRadius + 1]
    # ycoords, xcoords = np.mgrid[0: windowRadius*2+1, 0: windowRadius*2+1]
    Ix = ndimage.gaussian_filter(patch,scaleSigmaLevel, order = (0,1))
    Iy = ndimage.gaussian_filter(patch,scaleSigmaLevel, order = (1,0))
    mag = np.hypot(Ix,Iy)
    direction = np.arctan2(Iy,Ix)
    #should be able to index mag and direction
    # gaussianWeight = np.exp(-)
    weights = np.multiply(mag,np.exp(-((xcoords - windowRadius)**2 +\
                                    (ycoords- windowRadius)**2) /\
                                          2*(1.5 * scaleSigmaLevel)**2))
    
    plt.autoscale(False)
    plt.plot(fiveT[3],fiveT[2], 'ro')
plt.show()

# showMPL(original)
