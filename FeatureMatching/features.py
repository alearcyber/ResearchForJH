import cv2
import numpy as np
from matplotlib import pyplot as plt


"""
Utility Functions
"""
###########################################
# Print out attributes of a single keypoint
###########################################
def print_kp(keypoint: cv2.KeyPoint):
    title = 'Keypoint'
    attributes = [   # [ (name1, attribute_value1),...]
        ('classID', keypoint.class_id,),
        ('angle', keypoint.angle),
        ('octave', keypoint.octave),
        ('coords', keypoint.pt),
        ('size', keypoint.size),
        ('response', keypoint.response),
    ]
    out = title + '('
    for name, value in attributes:
        out += name + ':' + str(value) + ', '

    out = out[:-2] + ")"
    print(out)



###########################################
# Construct a Histogram
###########################################
def make_histogram(X, name):
    plt.hist(X, bins=50)
    plt.gca().set(title=name, ylabel='Frequency')
    plt.show()



##################################################
# convert list of opencv keypoints to numpy array
##################################################
def convert_kp_to_array(keypoints):
    #significant attributes for a keypoint:
    # [x, y, response, angle, size, octave]

    #construct new array
    out = []

    #iterate through keypoints and pull out attributes
    for kp in keypoints:
        x = kp.pt[0]
        y = kp.pt[1]
        response = kp.response
        angle = kp.angle
        size = kp.size
        octave = kp.octave

        #keypoints as a single list
        kp_as_list = [x, y, response, angle, size, octave]

        #try converting KP to integers
        kp_as_list = [int(e) for e in kp_as_list]

        if None in kp_as_list:
            print('FOUND A NONE')

        #append to output array
        out.append(kp_as_list)



    #convert to np array
    return np.array(out)




"""
Test Functions
"""
def test1():
    color = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/FeatureMatching/images/planedash.png")
    gray = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/FeatureMatching/images/planedash.png", 0)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)  #kp is tuple, des is numpy array

    print("kp len:", len(kp))
    print("type of first element in kp:", type(kp[0]))
    print("des shape:", des.shape)


    print("--Attributes of the first kp--")
    print_kp(kp[0])

    #output_image = cv2.drawKeypoints(color, kp, 0, (255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("keypoints", output_image)
    #cv2.waitKey()

    print(des)



####################################################################
# get statistics about the global descriptors of keypoints
####################################################################
def test2():
    #read in image
    color = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/FeatureMatching/images/planedash.png")
    gray = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/FeatureMatching/images/planedash.png", 0)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)  # kp is tuple, des is numpy array

    octaves = []
    sizes = []
    responses = []
    angles = []

    for kp in kps:
        octaves.append(kp.octave)
        sizes.append(kp.size)
        responses.append(kp.response)
        angles.append(kp.angle)



    make_histogram(octaves, 'Octave')
    make_histogram(sizes, 'Size')
    make_histogram(responses, 'Response')
    make_histogram(angles, 'Angles')






####################################################################
# match just the global keypoints, compare to local keypoints
####################################################################
def global_vs_local_matching():

    # All the images
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')


    # get descriptors
    sift = cv2.SIFT_create()
    og_global_desc, og_local_desc = sift.detectAndCompute(screenshot, None)
    new_global_desc, new_local_desc = sift.detectAndCompute(picture, None)

    # save pretransformed global features
    tuple_og_global_desc = og_global_desc[:]
    tuple_new_global_desc = new_global_desc[:]

    #transform/convert global descriptors to numpy arrays
    og_global_desc = convert_kp_to_array(og_global_desc)
    new_global_desc = convert_kp_to_array(new_global_desc)


    print('local shape:', og_global_desc.shape)
    print('local shape:', new_global_desc.shape)

    print('og shape:', og_local_desc.shape)
    print('og shape:', new_local_desc.shape)


    #print('1 global internals:', og_global_desc[0])
    #print('2 global internals:', new_global_desc[0])


    print('TEST HERE')
    print('old typings:', og_global_desc[0].shape)
    print('new typings:', new_global_desc[0].shape)

    #print("global type:", type(og_global_desc))
    #print('local type:', type(og_local_desc))
    #print('local shape:', og_local_desc.shape)

    #matching
    #matching types
    #  cv2.NORM_L1, cv2.NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    local_matches = bf.match(og_local_desc, new_local_desc)
    #global_matches = bf.match(og_global_desc, new_global_desc)



    #sort the matches
    local_matches = sorted(local_matches, key=lambda x: x.distance)
    #global_matches = sorted(global_matches, key=lambda x: x.distance)



    #draw matches
    local_out = cv2.drawMatches(screenshot, og_local_desc, picture, new_local_desc, local_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #global_outs = cv2.drawMatches(screenshot, og_global_desc, picture, new_global_desc, global_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


    plt.imshow(local_out)
    plt.show()



def match_global_keypoints():
    # All the images
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')

    # keypoints
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(screenshot, None)
    g2, l2 = sift.detectAndCompute(picture, None)

    #transform to array
    g1 = convert_kp_to_array(g1)
    g2 = convert_kp_to_array(g2)
    print(g2.shape)
    print(g1.shape)


    #matching
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    matches = bf.match(g1, g2)








def compare_types():
    # All the images
    screenshot = cv2.imread('images/planedash.png')
    picture = cv2.imread('images/dash2.png')
    obstructed = cv2.imread('images/dash3.png')

    # keypoints
    sift = cv2.SIFT_create()
    g1, l1 = sift.detectAndCompute(screenshot, None)
    g1 = convert_kp_to_array(g1)

    print("===FIRST THING===")
    print(l1[0].shape)
    print(g1[0].shape)







# Histogram Maximization
# seeing what transofmr to make the screenshot and the other thing similar
# blurring
# mayber only black and white



if __name__ == '__main__':
    #match_global_keypoints()
    compare_types()

