import numpy as np
import cv2
from keras.datasets import cifar10
from matplotlib import pyplot as plt
import os


def showMPL(image):
    plt.imshow(image, cmap = "gray", interpolation = "nearest")
    plt.show()


image1 =  cv2.imread("image_01.png")
image2 =  cv2.imread("image_02.png")