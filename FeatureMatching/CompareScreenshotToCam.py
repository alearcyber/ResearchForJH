"""
This file will begin to explore comparing the screenshot of an image to the camera.
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt



###############################
# grab histogram of an image
###############################
def getHist(img):
    return cv2.calcHist([img], [0], None, [256], [0, 256])



###############################
# plot histograms of an image
###############################
def plotHist(images):
    for img in images:
        plt.hist(img.ravel(), 256, [0, 256])
    plt.show()




"""
First thing here is to check if out-of-the-box methods can simply confirm if the view is obstructed.
This function will compare a screenshot to an image taken at the same time using a camera.
"""
def image_difference_test():
    #read images into memory
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')

    #find the difference between the screenshot and picture with subtraction
    #diff = cv2.absdiff(screenshot, picture)


    #comparing histrograms
    #images = [screenshot, picture, obstructed]
    #images = [screenshot]
    images = [picture]
    plotHist(images)



    #cv2.imshow("difference", diff)
    #cv2.waitKey()




if __name__ == '__main__':
    image_difference_test()







