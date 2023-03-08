import numpy as np
import cv2 as cv


# test image
# '/Users/aidanlear/Desktop/Research-Summer-2022/misc-pics/noisereduction.png'


def find_contours_test(im_path):
    """
    finds the contours in an image, this is what dr hauenstein and I came up with
    """
    im = cv.imread(im_path)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret_val, thresh = cv.threshold(imgray, 125, 255, 0)  # 50:50 thresh
    # contours, hierarchy = cv.findContours(imgray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    """
    cnt = contours[-1]
    print(type(cnt))
    print(type(contours[3]))
    cv.drawContours(im, [cnt], 0, (0,255,0), 3)
    """

    # options
    """
    RETR_CCOMP = 2
    RETR_EXTERNAL = 0
    RETR_FLOODFILL = 4
    RETR_LIST = 1
    RETR_TREE = 3
    """

    the_best_around = sorted(contours, key=cv.contourArea, reverse=True)
    cv.drawContours(im, the_best_around[0:10], -1, (0, 255, 0), 3)

    # cv.imshow('thresh', thresh)
    cv.imshow('thecountours', im)
    cv.waitKey()



def find_contours(im, n_contours):
    """
    finds the contours in an image
    To be used as a function that returns the image with contours drawn.
    Only draws the biggest contour right now
    """
    #process and find contours
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    ret_val, thresh = cv.threshold(imgray, 125, 255, 0)  # 50:50 thresh
    contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    the_best_around = sorted(contours, key=cv.contourArea, reverse=True)
    cv.drawContours(im, the_best_around[0:n_contours], -1, (0, 255, 0), 3)
    return im

def capture_video():
    # define a video capture object
    vid = cv.VideoCapture(0)

    while (True):

        # Capture the video frame
        # by frame
        ret, frame = vid.read()



        # Display the resulting frame
        cv.imshow('frame', find_contours(frame, 10))

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv.destroyAllWindows()



capture_video()








