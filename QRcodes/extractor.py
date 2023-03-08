"""
Holds the functionality for finding the screen and flattening it
"""

import cv2
import numpy as np
import PIL.Image, PIL.ImageDraw


def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                    [0, height - 1]], dtype="float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")
    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height)), ordered_corners


def perspective_transform_already_ordered(image, corners):

    # Order points in clockwise order
    top_l, top_r, bottom_r, bottom_l = corners

    #Log an image of where things are going to be saved
    img = PIL.Image.fromarray(image)
    draw = PIL.ImageDraw.Draw(img)
    radius = 5 # radius of the circles to be drawn, in pixels
    draw.ellipse((top_l[0] - radius, top_l[1] - radius, top_l[0] + radius, top_l[1] + radius), fill='red', outline='red')
    draw.ellipse((top_r[0] - radius, top_r[1] - radius, top_r[0] + radius, top_r[1] + radius), fill='red', outline='red')
    draw.ellipse((bottom_r[0] - radius, bottom_r[1] - radius, bottom_r[0] + radius, bottom_r[1] + radius), fill='red', outline='red')
    draw.ellipse((bottom_l[0] - radius, bottom_l[1] - radius, bottom_l[0] + radius, bottom_l[1] + radius), fill='red', outline='red')
    #config.log_image(img, 'fourcorners')

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1],
                           [0, height - 1]], dtype="float32")
    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))


def extract(image):
    """
    This function will attempt to find the border of the qr code grid

    :returns: Returns both the extracted image as well as the corners of the subimage extracted from the original
        image in a 2-tuple where the 0th index is the sub-image and the 1-index is the corners.
        The corners ordered topleft, topright, bottomright, bottomleft
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    """
    # original code; there is a loop for some reason
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        transformed = perspective_transform(original, approx)
        break
    """

    # my replacement; I just operate on the first element
    first_cnt = cnts[0]
    peri = cv2.arcLength(first_cnt, True)
    approx = cv2.approxPolyDP(first_cnt, 0.015 * peri, True)
    transformed, corners = perspective_transform(original, approx)

    #return
    return transformed, corners


def test_extract():
    """
        Test extracting the display
        Shows in a new windows what was extracted
    """
    image = cv2.imread(input("Image Path:"))
    transformed, corners = extract(image)
    cv2.imshow('transformed', transformed)
    cv2.waitKey(0)



if __name__ == '__main__':
    test_extract()




