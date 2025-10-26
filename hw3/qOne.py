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

def euclidean_distance(a,b):
    return np.linalg.norm(a-b)

    
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
    octaveIndex = 0 #which octave it is in (both DOG and Gaussian octave amounts should be the same)
    for octave in DOGScaleSpace:
        for i in range(1,len(octave)-1): # i = which image in the octave
            prev, curr, next_ = octave[i-1], octave[i], octave[i+1]
            scaleSigmaLevel = 1.6 * (1.26**i) * (2**octaveIndex)
            Dxx = ndimage.gaussian_filter(curr,scaleSigmaLevel, order = (0,2), mode = 'nearest')
            Dyy = ndimage.gaussian_filter(curr,scaleSigmaLevel, order = (2,0), mode = 'nearest')
            Dxy = ndimage.gaussian_filter(curr, scaleSigmaLevel, order = (1,1), mode = 'nearest')
            TrH = Dxx + Dyy
            DetH = Dxx*Dyy - Dxy**2
            for y in range(1, curr.shape[0]-1):
                for x in range(1, curr.shape[1]-1):
                    #3a and b
                    hessianRatio = TrH[y,x]**2 / (DetH[y,x] + 1e-12)
                    if DetH[y][x] <= 0 or hessianRatio > ((11**2)/10):
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


    #36 bin
    for fiveT in extrema:
        scaleSigmaLevel = fiveT[1][-1]
        windowRadius = int(round(3 * scaleSigmaLevel))
        #expand for boundary pixels:
        patch = gaussianOctaves[fiveT[1][0]][fiveT[1][1]][fiveT[1][2] - windowRadius -1 : fiveT[1][2] + windowRadius + 2,
                fiveT[1][3] - windowRadius-1 : fiveT[1][3] + windowRadius + 2]
        # if(patch.shape[0] != patch.shape[1]):continue
        # ycoords, xcoords = np.mgrid[0: windowRadius*2+3, 0: windowRadius*2+3]
        #dont expand:
        # patch = gaussianOctaves[fiveT[1][0]][fiveT[1][1]][fiveT[1][2] - windowRadius : fiveT[1][2] + windowRadius + 1,
        #           fiveT[1][3] - windowRadius : fiveT[1][3] + windowRadius + 1]
        # ycoords, xcoords = np.mgrid[0: windowRadius*2+2, 0: windowRadius*2+2]
        magnitudes = []
        directions = []
        weightedMag = []
        for i in range(1, patch.shape[0]-1):
            for j in range(1, patch.shape[1]-1):
                one = patch[i,j+1] - patch[i,j-1]
                two = patch[i+1,j] - patch[i-1,j]
                mag = np.sqrt(one**2 + two**2)
                direc = np.degrees(np.arctan2(two,one))
                magnitudes.append(mag)
                directions.append(direc)
                gaussianWeight = np.exp(-(((j - windowRadius)**2 + (i - windowRadius)**2) / (2 * (1.5 * scaleSigmaLevel)**2)))
                weightedMag.append(mag * gaussianWeight)
            
        directionsDegrees = np.degrees(directions)%360
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


    #8 bin
    for point in extrema:
        # ipdb.set_trace()
        img = gaussianOctaves[point[1][0]][point[1][1]] #which octave, which image in that octave
        ogX, ogY = point[1][3], point[1][2] #x, y
        # scale_factor = 2 ** point[1][0]

        # size = int(8 * scale_factor) 
        # patch = img[ogY - size:ogY + size+1, ogX - size:ogX + size+1]
        half_size = 8
        if ogY - half_size < 0 or ogY + half_size+1 >= img.shape[0]-1 \
        or ogX - half_size < 0 or ogX + half_size+1 >= img.shape[1]-1:
            continue
        patch = img[ogY - 9:ogY + 9, ogX-9: ogX+9]
        # if(patch.shape[0] != patch.shape[1]):continue
        # rotated = ndimage.rotate(patch, -point[1][-1], reshape = False, order = 1) #cval=np.mean(patch)) 
        rotPatch = ndimage.rotate(patch,-point[1][-1],reshape = False, order = 1)
        windowRadius = 7.5
        weightedMag = []
        directions = []
        for i in range(1,patch.shape[0]-1):
            magRow = []
            dirRow = []
            for j in range(1,patch.shape[1]-1):
                one = rotPatch[i,j+1] - rotPatch[ i,j-1]
                two = rotPatch[i+1,j] - rotPatch[i-1,j]
                mag = np.sqrt(one**2 + two**2)
                angle = (np.degrees(np.arctan2(two,one))+360) % 360
                dirRow.append(angle)
                gaussianWeight = np.exp(-(((j - windowRadius)**2 + (i - windowRadius)**2) / (2 * (1.5 * point[1][-2])**2)))
                magRow.append(mag*gaussianWeight)
            weightedMag.append(magRow)
            directions.append(dirRow)
        # ipdb.set_trace()
        length = max(map(len, weightedMag))
        weightedMag=np.array([xi+[0]*(length-len(xi)) for xi in weightedMag])
        length = max(map(len, directions))
        directions = np.array([xi + [0]* (length-len(xi)) for xi in directions])

        descriptors = []
        for i in range(4):
            for j in range(4):
                subregionMag = weightedMag[i*4:(i+1)*4,j*4:(j+1)*4]
                subregionDir = directions[i*4:(i+1)*4,j*4:(j+1)*4]
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
        if bruh == 0: firstPoints.append(point) #first image
        else: secondPoints.append(point) #second image

bestMatches = []

for firsts in firstPoints:
    best_dist = float('inf')
    best_second = None
    for seconds in secondPoints:
        out = euclidean_distance(firsts[1][-1], seconds[1][-1]) #descriptors of both points
        if out < best_dist:
            best_dist = out
            best_second = seconds
    if best_second is not None:
        points = [firsts,best_second]
        bestMatches.append((best_dist,points))
        secondPoints.remove(best_second)
bestMatches.sort(key = lambda x: x[0])


#at this point, point tuple looks like:
#(DOGresponse, [octaveIndex,scaleIndex,y,x,scaleSigmaLevel,angle, descriptor])

combined = np.hstack((displayOne,np.zeros((displayOne.shape[0],50,displayOne.shape[2]), dtype = displayOne.dtype), displayTwo))
offset_x = displayOne.shape[1] + 50  # width offset for right image

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
plt.imshow(combined)
plt.axis('off')
plt.show()
