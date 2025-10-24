import numpy as np
from matplotlib import pyplot as plt
import skimage as ski
import cv2
import os
from scipy import ndimage
from scipy.spatial.distance import cdist

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

def manhattan_distance(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"Arrays must have the same shape, got {a.shape} and {b.shape}")
    return np.sum(np.abs(a - b))


# Initialize the SIFT detector
# sift = cv2.SIFT_create()

# bgrA = cv2.imread("imgA.png")
# bgrB = cv2.imread("imgB.png")

# grayA = cv2.cvtColor(bgrA, cv2.COLOR_BGR2GRAY)
# grayB = cv2.cvtColor(bgrB, cv2.COLOR_BGR2GRAY)

# # Detect keypoints and compute descriptors for both images
# keypoints1, descriptors1 = sift.detectAndCompute(grayA, None)
# keypoints2, descriptors2 = sift.detectAndCompute(grayB, None)
# # Draw the keypoints on the images
# image1_keypoints = cv2.drawKeypoints(bgrA, keypoints1, None, color = (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# image2_keypoints = cv2.drawKeypoints(bgrB, keypoints2, None, color = (0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# cv2.imshow("window",image1_keypoints)
# cv2.waitKey(0)
# cv2.imshow("window",image2_keypoints)
# cv2.waitKey(0)
    
firstPoints = []
secondPoints = []
bestMatches = []

displayOne = []
displayTwo = []

for bruh in range(2):
    imageName = ""
    if bruh == 0: imageName = "imgA.png"
    else: imageName = "imgB.png"

    pathToBase = os.path.join(os.getcwd(),imageName) #path to greyscale kavli upperleft corner

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
            # scaleSigmaLevel = 1.6 * (1.26**(octaveIndex * i ))
            scaleSigmaLevel = 1.6 * (1.26**i) * (2**octaveIndex)
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
                        extrema.append((currentValue,[octaveIndex,i,y,x,scaleSigmaLevel]))
                    elif currentValue <= np.min(flat) + 1e-6 and np.abs(currentValue) >= 0.03:
                        extrema.append((np.abs(currentValue),[octaveIndex,i,y,x,scaleSigmaLevel]))
        octaveIndex +=1
    # extrema.sort(reverse=True)
    # extrema = extrema[:100] 

    #4
    #a
    secondaryExtrema = []
    for fiveT in extrema:
        scaleSigmaLevel = fiveT[1][-1]
        windowRadius = int(round(3 * scaleSigmaLevel))
        #expand for boundary pixels:
        patch = gaussianOctaves[fiveT[1][0]][fiveT[1][1]][fiveT[1][2] - windowRadius -1 : fiveT[1][2] + windowRadius + 2,
                fiveT[1][3] - windowRadius-1 : fiveT[1][3] + windowRadius + 2]
        if(patch.shape[0] != patch.shape[1]):continue
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
        weightedMag = np.multiply(mag,np.exp(-((xcoords - windowRadius)**2 +\
                                        (ycoords- windowRadius)**2) /\
                                            2*(1.5 * scaleSigmaLevel)**2))
        directionsDegrees = np.degrees(direction)%360
        hist, binEdges = np.histogram(directionsDegrees, bins =36, range =(0,360), weights = weightedMag)
        binEdges = binEdges[:-1]
        maxBinValue = np.max(hist)
        dominantAngle = binEdges[np.argmax(hist)]
        fiveT[1].append(dominantAngle)
        threshold = 0.8 * maxBinValue
        secondaryAngles = binEdges[threshold <= hist]
        for i in range(len(secondaryAngles)):
            list = fiveT[1][:-1]
            list.append(secondaryAngles[i])
            secondaryExtrema.append((fiveT[0],list))

    extrema.extend(secondaryExtrema)
    display_img = original
    display_img =255 - (display_img * 255).astype(np.uint8)
    if bruh == 0: displayOne = display_img
    else: displayTwo = display_img
    for point in extrema:
        img = gaussianOctaves[point[1][0]][point[1][1]]
        # ogX, ogY = point[1][3] * 2 ** point[1][0], point[1][2] * 2 **point[1][0]
        ogX, ogY = point[1][3], point[1][2]
        # patch = img[ogY-7:ogY+9, ogX-7:ogX+9]
        scale_factor = 2 ** point[1][0]
        size = int(8 * scale_factor)
        patch = img[ogY - size:ogY + size, ogX - size:ogX + size]
        if(patch.shape[0] != patch.shape[1]):continue
        rotated = ndimage.rotate(patch, -point[1][-1], reshape = False, order = 1) #cval=np.mean(patch))
        Ix = ndimage.sobel(rotated, axis = 1)
        Iy = ndimage.sobel(rotated, axis = 0)
        mag = np.hypot(Ix,Iy)
        ycoords,xcoords = np.mgrid[0:16, 0:16]
        directions = (np.degrees(np.arctan2(Ix,Iy))) % 360
        windowRadius = 7.5  # center of the 16x16 patch
        gaussianWeight = np.exp(-((xcoords - windowRadius)**2 + (ycoords - windowRadius)**2) /
                                (2 * (1.5 * point[1][-2])**2))
        if mag.shape != gaussianWeight.shape: continue
        weightedMag = np.multiply(mag, gaussianWeight)
        descriptors = []
        for i in range(4):
            for j in range(4):
                subregionMag = weightedMag[i*4:(i+1)*4, (j)*4:(j+1)*4]
                subregionDir = directions[i*4:(i+1)*4, (j)*4:(j+1)*4]
                subregionHist,subregionBin = np.histogram(subregionDir, bins = 8, range = (0,360), weights = subregionMag)
                descriptors.extend(subregionHist)
        descriptors = np.array(descriptors)
        norm = np.linalg.norm(descriptors)
        if norm > 1e-6:
            descriptors/=norm
        else:
            descriptors[:] = 0
        descriptors = np.clip(descriptors, 0, 0.2)
        norm = np.linalg.norm(descriptors)
        if norm > 1e-6:
            descriptors /= norm
        point[1].append(descriptors)
        if bruh == 0: firstPoints.append(point)
        else: secondPoints.append(point)

for firsts in firstPoints:
    best = (900000, 0)
    for seconds in secondPoints:
        out = manhattan_distance(firsts[1][-1], seconds[1][-1]) #descriptors
        if out < best[0]:
            best = (out,seconds)
    points = [firsts,best[1]]
    bestMatches.append((out,points))
bestMatches.sort(key = lambda x: x[0])

#at this point, point tuple looks like:
#(DOGresponse, [octaveIndex,scaleIndex,y,x,scaleSigmaLevel,angle, descriptor])

combined = np.hstack((displayOne,np.zeros((displayOne.shape[0],50,displayOne.shape[2]), dtype = displayOne.dtype), displayTwo))
offset_x = displayOne.shape[1]  # width offset for right image

# Draw top N matches
for (dist, (kp1, kp2)) in bestMatches[:30]:
    # Compute pixel positions (accounting for octave scaling)
    x1 = kp1[1][3] * (2 ** kp1[1][0])
    y1 = kp1[1][2] * (2 ** kp1[1][0])
    x2 = kp2[1][3] * (2 ** kp2[1][0]) + offset_x
    y2 = kp2[1][2] * (2 ** kp2[1][0])

    color = tuple(np.random.randint(0, 255, 3).tolist())
    cv2.circle(combined, (int(x1), int(y1)), 4, color, -1)
    cv2.circle(combined, (int(x2), int(y2)), 4, color, -1)
    cv2.line(combined, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

plt.figure(figsize=(14, 8))
plt.imshow(combined)#cv2.cvtColor(combined, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# for i in range(len(bestMatches)):
#     ogX1, ogY1 = bestMatches[i][1][0][1][3] * 2 ** bestMatches[i][1][0][1][0], bestMatches[i][1][0][1][2] * 2 **bestMatches[i][1][0][1][0]
#     ogX2, ogY2 = bestMatches[i][1][1][1][3] * 2 ** bestMatches[i][1][1][1][0], bestMatches[i][1][1][1][2] * 2 **bestMatches[i][1][1][1][0]
#     # ogX1, ogY1 = bestMatches[i][1][0][1][3], bestMatches[i][1][0][1][2]
#     # ogX2, ogY2 = bestMatches[i][1][1][1][3], bestMatches[i][1][1][1][2]
#     rr, cc = ski.draw.circle_perimeter(int(round(ogY1)), int(round(ogX1)), (int(round(bestMatches[i][1][0][1][-3])*2.75))+10, shape=original.shape)
#     match bestMatches[i][1][0][1][0]:
#         case 0: displayOne[rr,cc] = [0,255,0] #green
#         case 1: displayOne[rr,cc] = [0,0,255] #blue
#         case 2: displayOne[rr,cc] = [255,255,0] #yellow
#     rr, cc = ski.draw.circle_perimeter(int(round(ogY2)), int(round(ogX2)), (int(round(bestMatches[i][1][1][1][-3])))+10, shape=original.shape)
#     match bestMatches[i][1][1][1][0]:
#         case 0: displayTwo[rr,cc] = [0,255,0] #green
#         case 1: displayTwo[rr,cc] = [0,0,255] #blue
#         case 2: displayTwo[rr,cc] = [255,255,0] #yellow
# showMPL(displayOne)
# showMPL(displayTwo)



    #compute manhattan distances. 10-15 smallest distances are keypoint matches
