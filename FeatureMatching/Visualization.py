import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import cv2
import numpy as np
import math


##########################################################################
# return list of n colors that advance along the rainbow. (B, G, R)
##########################################################################
def colorize(n):
    colors = [(211, 0, 148), (130, 0, 75), (255, 0, 0), (0, 255, 0), (0, 255, 255), (0, 127, 255), (0, 0, 255)] #BGR

    #assign a unique color to each if input is small enough
    if n <= len(colors):
        return colors[:n]

    #increase n so it is divisible by 6
    #n_ = n + 6 - (n % 6)

    b = (n + 1)//6  #amount in each group of 6
    result = []
    for i in range(6):
        result += colorize_h(colors[i], colors[i+1], int(math.log2(b + 1) + 1))

    return result[:n]





##################################################
#Inner method for colorizing
##################################################
def colorize_h(left, right, level):
    color = ((left[0] + right[0])//2, (left[1] + right[1])//2, (left[2] + right[2])//2) # midpoint color
    if level <= 1: #base case
        return [color]
    else:  #general case
        return colorize_h(left, color, level-1) + [color] + colorize_h(color, right, level-1)






###############################################
#Make elbow plot
###############################################
def elbow(X, k_range, title=None):
    # perform the pre-clustering
    cost = []
    for i in k_range:
        km = KMeans(n_clusters=i, max_iter=500)
        km.fit(X)
        cost.append(km.inertia_)

    # make the elbow plot
    plt.plot(k_range, cost, color='b')
    plt.xlabel('Value of K')
    plt.ylabel('Squared Error (Cost)')
    if not(title is None):
        plt.title(title)
    plt.show()




##################################################
#Draw the match for 2 given keypoints in an image
##################################################
def draw_matches(A, B, img1, img2, mode='horizontal', thickness=2):
    n = len(A)
    assert n == len(B)
    out = np.concatenate((img1, img2), axis=(1 if mode == 'horizontal' else 0))
    colors = colorize(n)
    x_offset = img1.shape[1]
    y_offset = img1.shape[0]
    for i in range(n):
        cv2.line(out, A[i], (B[i][0]+x_offset, B[i][1]), colors[i], thickness)

    cv2.imshow('Matches', out)
    cv2.waitKey()










def combine_images_test():
    i1 = cv2.imread('images/dash2.png')
    i2 = cv2.imread('images/dash3.png')

    i3 = np.concatenate((i1, i2), axis=1)

    height = i3.shape[0]
    width = i3.shape[1]

    cv2.line(i3, (0, 0), (width, height), (0, 0, 255), 2)

    colors = colorize(200)
    x = 0
    y = height//2
    for color in colors:
        cv2.circle(i3, (x, y), radius=4, color=color, thickness=-1)
        x += 5


    #cv2.circle(i3, (width//2, height//2), radius=100, color=(130, 0, 75), thickness=-1)


    cv2.imshow('Combined', i3)
    cv2.waitKey()




def test():
    # combine_images_test()
    p1 = [
        (200, 200),
        (400, 400),
        (600, 600)
    ]
    p2 = [
        (200, 220),
        (400, 420),
        (600, 620)
    ]
    draw_matches(p1, p2, cv2.imread('images/dash2.png'), cv2.imread('images/dash3.png'))




