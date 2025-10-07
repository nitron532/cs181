import numpy as np
import skimage as ski
import matplotlib
import matplotlib.pyplot as plt
import os

def downSampleByHalf(image):
    imageHalf = np.delete(image, list(range(1,image.shape[0],2)),axis = 1)
    imageHalfHalf = np.delete(imageHalf, list(range(1, image.shape[1],2)),axis = 0)
    return imageHalfHalf

def gaussianAndList(list):
    gaussianCurrentLevel = downSampleByHalf(ski.filters.gaussian(list[-1]))
    list.append(gaussianCurrentLevel)

pathToBase = os.path.join(os.getcwd(),"upperleftcorner.png") #path to greyscale kavli upperleft corner

baseLevel = ski.io.imread(pathToBase) #read image into baseLevel variable

gaussianList = [] #to store gaussian pyramid images

gaussianLevel0 = ski.filters.gaussian(baseLevel) #512x512 gaussianed image, sigma = 1

gaussianList.append(gaussianLevel0) #add to list

for i in range(0, 4):#generate downsampled and gaussianed from 512 -> 32 halfed each time
    gaussianAndList(gaussianList)


for i in gaussianList: #display using matplotlib so i can download them
    ski.io.imshow(i)
    print(i.shape)
    plt.show()
