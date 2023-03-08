"""
DESCRIPTION
- place output photos in same location as the script
- I want this to simply take the photos and bring them to another place.
- Open the camera
- Highlight the contours
- take the thing and do stuff with it
"""
import sys
import cv2
import numpy as np
import os
import time




def reorder(corners):
    """
    Given a list of out of order corners, place them in the following order:
    TL, TR, BR, BL
    """
    result = sorted(corners, key=lambda corner: corner[0] * corner[1])  # sort on the product
    if result[1][0] < result[2][0]:  # check the TR and BL placement, TR should have bigger x value
        result[1], result[2] = result[2], result[1]  # readjust TR and BL placement
    result[2], result[3] = result[3], result[2]
    return result



def perspective_transform(image, corners):
    #is commented in calibrate2.py
    top_l, top_r, bottom_r, bottom_l = corners
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dimensions)
    return cv2.warpPerspective(image, matrix, (width, height))



def find_best_contour(im):
    """
    finds the contours on the image.
    This function will ALSO return the contour objects themselves, so they can be used to crop thru homography.
    """
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  #convert image to grayscale
    ret_val, threshed_image = cv2.threshold(imgray, 125, 255, 0)  # 50:50 thresh
    contours, hierarchy = cv2.findContours(threshed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find the contours
    the_best_around = sorted(contours, key=cv2.contourArea, reverse=True) # sort the images by largest contour first

    # finding only contours that approximate to four corners
    n = 5 # number of contours to test
    for contour in the_best_around[:n]:
        peri = cv2.arcLength(contour, True)
        corners = cv2.approxPolyDP(contour, 0.04 * peri, True)  # ORDER: top left, bottom left, bottom right, top right
        if corners.shape[0] == 4:
            cv2.drawContours(im, [contour], -1, (0, 255, 255), 3)
            corners = [(e[0, 0], e[0, 1]) for e in corners]  # fix the weird formatting
            reordered_corners = np.float32(reorder(corners))  # reorder the corners here
            return im, reordered_corners
    return im, []







def main_loop():
    #find the camera
    vid = None
    for i in range(3):
        vid = cv2.VideoCapture(i)
        if vid.isOpened():
            break
    assert not(vid is None), "ERROR: Could not find video feed."


    #video feed loop
    photos = []  # cache the images taken
    while True:
        ret, frame = vid.read()
        if not ret:
            print("Camera feed has been closed or interrupted.")
            break
        original = frame.copy()
        image_cont_drawn, reordered_corners = find_best_contour(frame)
        cv2.imshow(f'Select Contour with SPACE, confirm Selection with ENTER.', frame)  # contours are found and displayed hear
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):  # quit on pressing q
            break
        elif key_pressed == ord('\r'):  # enter pressed
            suspect = perspective_transform(original, reordered_corners)
            cv2.imshow('extracted image', suspect)
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            photos.append((suspect, timestamp))


    #save the cache
    out_dir = r'/Users/aidanlear/Desktop/PHOTOS-TAKEN'
    try:
        os.mkdir(out_dir)
    except FileExistsError:
        pass #dont need to handle, it already exists

    for photo, timestamp in photos:
        cv2.imwrite(out_dir + '/' + timestamp + '.png', photo)



if __name__ == '__main__':
    main_loop()

