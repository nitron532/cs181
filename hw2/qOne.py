import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
from harrisCorner import harrisCorner
import cv2
from scipy import ndimage

def downSampleByHalf(image):
    return image[::2,::2]

def showMPL(image):
    plt.imshow(image, cmap = "gray", interpolation = "nearest")
    # plt.axis("off")
    plt.show()

def showCV(image):
    cv2.imshow("", image)
    cv2.waitKey(0)

def showMPLDots(rgb,x,y):
    plt.imshow(rgb)
    plt.autoscale(False)
    plt.plot(x,y, 'ro')
    plt.show()

def topCoords(compute, thresh):
    threshold = thresh
    #find local maxima and minima
    localMaxima = ndimage.maximum_filter(compute, size = 5)
    #bool mask of trues wherever a max is found
    maximaMask = localMaxima == compute
    localMinima = ndimage.minimum_filter(compute, size = 5)
    #only responses above threshold
    diff = ((localMaxima-localMinima) > threshold)
    maximaMask[~diff] = False
    #find and label detected areas
    labeled, numObjects = ndimage.label(maximaMask)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return (x,y)

def logWithFeatures(image, sigma, size=None):
    # Generate LoG kernel
    if size is None:
        size = int(6 * sigma + 1) if sigma >= 1 else 7
    
    x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
    kernel = -(1/(np.pi * sigma**4)) * (1 - ((x**2 + y**2) / (2 * sigma**2))) * np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))

    result = ndimage.convolve(image, kernel)

    #values after LOG are small, so rescale them to fit threshold value
    compute = (result - result.min()) / (result.max() - result.min()) * 255
    x,y = topCoords(compute, 0.125)
    return (result,x,y)

# #1.
# blackSquareSm = np.ones((100,100))
# blackSquareSm[50:80,50:80] = 0
# toRotate = blackSquareSm
# showMPL(blackSquareSm)
blackSquares = np.ones((512,512))
blackSquares[60:100,60:100] = 0
blackSquares[300:380,180:260] = 0
blackSquares[100:220,300:420] = 0
blackSquares[340:480,340:480] = 0
# showMPL(blackSquares)

# #2.
sigmaList = [1,2,4,8]
# # for i in sigmaList:
#     # harrisCorner(blackSquareSm, i, 4,1)

# for i in sigmaList:
#     result, x,y = logWithFeatures(blackSquareSm, i)
#     # scale values to preserve the original LOG image result but with RGB channel added
#     grayNorm = (result - result.min()) / (result.max()-result.min() + 1e-8)
#     rgb = np.stack((grayNorm,)*3, axis = -1)
#     showMPLDots(rgb,x,y)

# for i in sigmaList:
#     #DOG by subtracting two diff gaussians of diff sigmas by factor k
#     g1 = ndimage.gaussian_filter(blackSquareSm, i)
#     g2 = ndimage.gaussian_filter(blackSquareSm, 1.4*i)
#     result = g2-g1;
#     compute = (result - result.min()) / (result.max() - result.min()) * 255
#     x,y = topCoords(compute, 0.1)
#     grayNorm = (result - result.min()) / (result.max()-result.min() + 1e-8)
#     rgb = np.stack((grayNorm,)*3, axis = -1)
#     showMPLDots(rgb,x,y)

# #3
# toRotate= ndimage.rotate(toRotate, angle = 30, reshape = False,cval = 1, order = 0)
# showMPL(toRotate)

# for i in sigmaList:
#     harrisCorner(toRotate, i, 4,1)

# for i in sigmaList:
#     result, x,y = logWithFeatures(toRotate, i)
#     # scale values to preserve the original LOG image result but with RGB channel added
#     grayNorm = (result - result.min()) / (result.max()-result.min() + 1e-8)
#     rgb = np.stack((grayNorm,)*3, axis = -1)
#     showMPLDots(rgb,x,y)

# for i in sigmaList:
#     #DOG by subtracting two diff gaussians of diff sigmas by factor k
#     g1 = ndimage.gaussian_filter(toRotate, i)
#     g2 = ndimage.gaussian_filter(toRotate, 1.4*i)
#     result = g2-g1;
#     compute = (result - result.min()) / (result.max() - result.min()) * 255
#     x,y = topCoords(compute, 0.1)
#     grayNorm = (result - result.min()) / (result.max()-result.min() + 1e-8)
#     rgb = np.stack((grayNorm,)*3, axis = -1)
#     showMPLDots(rgb,x,y)


#4
#a
bsDown = downSampleByHalf(blackSquares)
gaussians = []
downSampledGaussians = [blackSquares]
for i in sigmaList:
    g = ndimage.gaussian_filter(blackSquares, i)
    gaussians.append(g)
    blurred = ndimage.gaussian_filter(bsDown, i)
    downSampledGaussians.append(blurred) 
    bsDown = downSampleByHalf(blurred) 

# for i in gaussians:
#     showMPL(i)
# for i in downSampledGaussians:
#     showMPL(i)
index = 0
# for i in gaussians:
#     harrisCorner(i,-1,16,4)
# for i in downSampledGaussians:
#     if index >= 2:
#         if index == 4:
#             harrisCorner(i,-1,4,1)
#         else:
#             harrisCorner(i,-1,16,1)
#     else:
#         harrisCorner(i,-1,16,3)
#     index+=1

#c
for i in range(1,5):
    logged, x,y = logWithFeatures(downSampledGaussians[i], sigmaList[i-1])
    #k = 1.5 
    display_img = np.stack([logged]*3, axis=-1)     # shape (h,w,3)
    display_img = (display_img*255).astype(np.uint8) # convert 0-1 float to uint8
    #add the *255 for black square problem
# Draw red circles at corners
    for xcoord,ycoord in zip(x,y):
        rr, cc = ski.draw.circle_perimeter(int(xcoord), int(ycoord), sigmaList[i-1], shape=logged.shape)
        display_img[rr, cc] = [255, 0, 0]  # red (BGR if OpenCV, RGB here)
    showMPL(display_img)

