import numpy as np
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
import os

def downSampleByHalf(image):
    return image[::2,::2]

def gaussianAndList(list):
    gaussianCurrentLevel = downSampleByHalf(ski.filters.gaussian(list[-1])) #downsample a gaussian convoluted image
    list.append(gaussianCurrentLevel) #add it to the list

def laplacianAndList(laplacianList, gaussianList, index):
    laplacianCurrentLevel = np.subtract(gaussianList[index], ski.filters.gaussian(gaussianList[index], sigma = 1))
    #subtract one level of gaussian pyramid with the gaussian convoluted version of itself
    laplacianList.append(laplacianCurrentLevel)

pathToBase = os.path.join(os.getcwd(),"upperleftcorner.png") #path to greyscale kavli upperleft corner

baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable

baseLevel = baseLevel[:, :, 0] #remove any rgba value
baseLevel = baseLevel.astype(np.double) #ensure double values

gaussianList = [] #to store gaussian pyramid images

laplacianList = [] #to store laplacian pyramid images

gaussianLevel0 = ski.filters.gaussian(baseLevel, sigma = 1) #512x512 gaussianed image, sigma = 1

laplacianLevel0 = np.subtract(baseLevel, gaussianLevel0)

laplacianList.append(laplacianLevel0) 

gaussianList.append(gaussianLevel0) 

for i in range(0,4):#generate downsampled and gaussianed from 512 -> 32 halfed each time
    gaussianAndList(gaussianList)

for i in range(0,5):
    laplacianAndList(laplacianList,gaussianList, i)

dpi = 100

for i in gaussianList: #display using matplotlib so i can download them
    #configure matplotlib to show only the image in its true size and no plot borders 
    height, width = i.shape[:2]
    figsize = width / dpi, height / dpi
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(i, cmap = 'gray', interpolation = 'nearest')
    plt.axis("off")
    plt.tight_layout(pad = 0)
    #label and show the image
    print(i.shape, "gaussian")
    plt.show()


for i in laplacianList:
    #configure matplotlib to show only the image in its true size and no plot borders 
    height, width = i.shape[:2]
    figsize = width / dpi, height / dpi
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(i, cmap = 'gray', interpolation = 'nearest')
    plt.axis("off")
    plt.tight_layout(pad = 0)
    #label and show the image
    print(i.shape, "laplacian")
    plt.show()