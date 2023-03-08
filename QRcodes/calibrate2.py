"""
Stand Alone Calibrating sequence
"""

import cv2
import numpy
import itertools
import qrcode2


"""
-------------UTILITY FUNCTIONS--------------
"""
def find_camera(n=None):
    """
    returns the camera capture object itself.
    Cycles thru and finds the camera
    """
    if not(n is None):
        return cv2.VideoCapture(n)

    max_tested = 10 # max number of cameras to test
    for i in range(max_tested):
        showing_video = True
        video_feed = cv2.VideoCapture(i)
        camera_found = video_feed.isOpened()
        if not camera_found:
            showing_video = False
        while showing_video:
            ret, frame = video_feed.read()
            cv2.imshow(f'Camera {i}, Press Q for next, press Enter to select.', frame)
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == ord('q'):  # go next
                showing_video = False

            if key_pressed == ord('\r'): # return
                cv2.destroyAllWindows()
                return video_feed

        video_feed.release()
        cv2.destroyAllWindows()




def find_best_contour_with_details(im, selected=0):
    """
    finds the contours on the image.
    This function will ALSO return the contour objects themselves, so they can be used to crop thru homography.
    """
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  #convert image to grayscale
    ret_val, threshed_image = cv2.threshold(imgray, 125, 255, 0)  # 50:50 thresh
    contours, hierarchy = cv2.findContours(threshed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find the contours
    the_best_around = sorted(contours, key=cv2.contourArea, reverse=True) # sort the images by largest contour first
    try:
        the_best_around = the_best_around[0:4]
    except:
        pass
    for i in range(len(the_best_around)):
        if i == (selected % len(the_best_around)):
            color = (255, 0, 0) # blue
        else:
            color = (0, 255, 255)  # yellow
        cv2.drawContours(im, [the_best_around[i]], -1, color, 3)  # draw the contours on the new image
    return im, the_best_around  # return the image with contours draw on it



def reorder(corners):
    """
    Given a list of out of order corners, place them in the following order:
    TL, TR, BR, BL
    """
    result = sorted(corners, key=lambda corner: corner[0] * corner[1])  # sort on the product
    if result[1][0] < result[2][0]:  # check the TR and BL placement, TR should have bigger x value
        result[1], result[2] = result[2], result[1]  # readjust TR and BL placement
    result[2], result[3] = result[3], result[2]
    print(result)
    return result




def perspective_transform_already_ordered(image, corners):

    # Order points in clockwise order
    top_l, top_r, bottom_r, bottom_l = corners

    # Determine width of new image which is the max distance between
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = numpy.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = numpy.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = numpy.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = numpy.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = numpy.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))





def calibration_preprocess(image, n, corners):
    """
    Determine preprocessing parameters for the given image.
    See flowchart for specifics.
    Return values
        -1 -> error
        0 -> raw image
        1 -> just curve the image
        2 -> curve + sharpen
    """
    print('Pre-Processing...')
    # try raw
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # gray scale version, cropped
    raw_result = qrcode2.verify_image_percent(gray, n)
    if raw_result == 100:
        print('\tRaw Image scanned Perfectly.')
        write_results(0, corners)
        return 0

    elif raw_result > 60:
        print(f'\tRaw result is close({raw_result}), moving to curve...')
        image = apply_color_curve(image) # apply the color curve
        curved_results = qrcode2.verify_image_percent(image, n)
        if curved_results == 100:
            print('\tCurved scanned Perfectly.')
            write_results(1, corners)
        else:
            print(f'\tCurved results close ({curved_results}), moving to simple sharpening...')
            kernel_values = [1, 3]
            sigma_values = [1.0, 2.0, 3.0]
            amount_values = [0.2, 0.6, 1.0, 1.4, 1.8, 2.0]
            for _k, _s, _a in itertools.product(kernel_values, sigma_values, amount_values):
                sharpened_image = sharpen(gray, kernel_size=(_k, _k), sigma=_s, amount=_a)
                sharpened_results = qrcode2.verify_image(sharpened_image, 5)
                if sharpened_results == 100:
                    print(f'\tSuccess on simple sharpen. Values: kernel={_k}, sigma={_s}, amount={_a}')
                    write_results(2, corners, kernel=_k, sigma=_s, amount=_a)
                    return 2

            # if we made it to here, nothing worked, I suspect it will never get to this point
            #This is where I would add something to do a more detailed analysis of what parts of the screen failed
            print('Failed, exiting')
            return -1


    else:
        print(f'\tERROR. The raw image was very bad ({raw_result}). Exiting.')
        return -1




def write_results(result, corners, kernel=3, sigma=1, amount=1):
    """
    Write the results to calibrate.txt
    Should be in same directory as this script.
    The corners should be a numpy array
    File format
        result
        x1,y1;x2,y2;x3,y3;x4,y4
        kernel,sigma,amount
    """
    file = open('calibrate.txt', 'w')
    file.write(str(result) + '\n')
    line_with_corners = f'{corners[0, 0]},{corners[0, 1]};{corners[1, 0]},{corners[1, 1]};{corners[2, 0]},{corners[2, 1]};{corners[3, 0]},{corners[3, 1]}\n'
    file.write(line_with_corners)
    file.write(f'{kernel},{sigma},{amount}')
    file.close()




def apply_color_curve(image):
    """
    Apply the color curve like in GIMP
    """
    lut_in = [0, 127, 200]  # right indent?
    lut_out = [0, 52, 255]  # left indent?
    lut_8u = numpy.interp(numpy.arange(0, 256), lut_in, lut_out).astype(numpy.uint8)
    image_contrasted = cv2.LUT(image, lut_8u)
    return image_contrasted



def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """
    Return a sharpened version of the image, using an unsharp mask.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
    sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
    sharpened = sharpened.round().astype(numpy.uint8)
    if threshold > 0:
        low_contrast_mask = numpy.absolute(image - blurred) < threshold
        numpy.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



"""
----------------------------------------DRIVER FUNCTIONS----------------------------------------
"""

def calibrate():
    """
    main calibration function
    """
    vid = find_camera()
    selected = 0  # which square is being chosen
    while True:
        ret, frame = vid.read()
        original = frame.copy()
        # find the contours
        image_cont_drawn, contours = find_best_contour_with_details(frame, selected=selected)
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.04 * peri, True)
            cv2.polylines(frame, [corners], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow(f'Select Contour with SPACE, confirm Selection with ENTER.', frame)  # contours are found and displayed hear
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):  # quit on pressing q
            break
        elif key_pressed == 32: # space key
            selected += 1
        elif key_pressed == ord('\r'):  # enter pressed
            # Extract corners from the selected contour, transform perspective
            contour = contours[selected % len(contours)]
            peri = cv2.arcLength(contour, True)
            corners = cv2.approxPolyDP(contour, 0.04 * peri, True) #ORDER: top left, bottom left, bottom right, top right
            corners = [(e[0, 0], e[0, 1]) for e in corners]  # fix the weird formatting
            reordered_corners = numpy.float32(reorder(corners))  # reorder the corners here
            suspect = perspective_transform_already_ordered(original, reordered_corners)
            results = calibration_preprocess(suspect, 5, reordered_corners)
            print('calibration preprocess results:', results)
            cv2.imshow('extracted image', suspect)
            cv2.waitKey()


    vid.release()
    cv2.destroyAllWindows()





if __name__ == '__main__':
    calibrate()
