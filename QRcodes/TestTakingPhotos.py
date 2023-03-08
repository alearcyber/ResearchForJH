"""
"""
import cv2
import tkinter as tk
import PIL.Image
import PIL.ImageTk
import numpy
import extractor
import qrcode2
import thresh
import itertools
import PIL.Image, PIL.ImageDraw


"""
-----TESTS------
"""

def cam_test_1():
    # capture from camera at location 0
    camera = cv2.VideoCapture(0)


    print('backend name:', camera.getBackendName())

    print('Default Exposure:', camera.get(cv2.CAP_PROP_EXPOSURE))

    #take image and save
    ret, frame = camera.read()
    cv2.imwrite('exposure-default.png', frame)


    # set the width and height, and UNSUCCESSFULLY set the exposure time
    #camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    #camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)
    camera.set(cv2.CAP_PROP_EXPOSURE, 3)
    print('exposure-3:', camera.get(cv2.CAP_PROP_EXPOSURE))



    #take another image and save
    ret, frame = camera.read()
    cv2.imwrite('exposure-3.png', frame)


    #grab the focus of the image
    print('default Focus', camera.get(cv2.CAP_PROP_EXPOSURE))


    #set the focus to something else
    is_focus_supported = camera.set(cv2.CAP_PROP_FOCUS, 1)
    print('Is focus supported', is_focus_supported)


    #take another image and save
    ret, frame = camera.read()
    cv2.imwrite('focus-1.png', frame)




def test2():
    """ Figure out the camera backends and what is valid """



    #backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_INTEL_MFX, cv2.CAP_AVFOUNDATION, cv2.CAP_CV_IMAGES, cv2.CAP_CV_MJPEG, cv2.CAP_UEYE]
    backends = [(cv2.CAP_FFMPEG, 'CAP_FFMPEG'),
                (cv2.CAP_GSTREAMER, 'CAP_GSTREAMER'),
                (cv2.CAP_INTEL_MFX, 'CAP_INTEL_MFX'),
                (cv2.CAP_AVFOUNDATION,'CAP_AVFOUNDATION'),
                (cv2.CAP_UEYE, 'CAP_UEYE'),
                ]

    properties = [(cv2.CAP_PROP_MODE, 'CAP_PROP_MODE'),
                  (cv2.CAP_PROP_BRIGHTNESS, 'CAP_PROP_BRIGHTNESS'),
                  (cv2.CAP_PROP_EXPOSURE, 'CAP_PROP_EXPOSURE'),
                  (cv2.CAP_PROP_SHARPNESS, 'CAP_PROP_SHARPNESS'),
                  (cv2.CAP_PROP_FOCUS, 'CAP_PROP_FOCUS '),
                  (cv2.CAP_PROP_ZOOM, 'CAP_PROP_ZOOM '),
                  (cv2.CAP_PROP_ISO_SPEED, 'CAP_PROP_ISO_SPEED'),
                  ]
    availableBackends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
    print('available backends:', availableBackends)
    for backend, name in backends:
        #print out a section header for what backend is being used
        print(f'\n--{name}--')

        #instantiante the camera object with the background
        camera = cv2.VideoCapture(0, backend)

        #iterate over the properties
        for prop, prop_name in properties:
            is_property_accessible = camera.set(prop, camera.get(prop))
            #is_property_accessible = camera.set(prop, 1)
            print(f'\t{prop_name}: {is_property_accessible}')
            #print(f'\t{prop_name}: {camera.get(prop)}')

        #release the camera
        camera.release()



def check_color_curves():
    """points 4 0.20560747663551401 0 0.74454828660436134 1"""
    image = cv2.imread('/Users/aidanlear/Desktop/pasta_stand_dark_original.jpg')

    # first calculate the alpha and beta
    indent_left, indent_right = 52, 94   # indents from the color curve thing in gimp, 'V4L2', 
    xdist = indent_right - indent_left
    alpha = 256.0/float(xdist)
    beta = int(-1 * indent_left * alpha)

    # perform conversion
    new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # show the image
    cv2.imshow('new image', new_image)
    cv2.waitKey()

def color_curve_lut():
    """
    Described here: https://stackoverflow.com/questions/59851952/is-there-any-way-to-manipulate-an-image-using-a-color-curve-in-python
    """
    image = cv2.imread('/Users/aidanlear/Desktop/pasta_stand_dark_original.jpg')

    lut_in = [30, 127, 200]  # right indent?
    lut_out = [0, 52, 255]  # left indent?

    lut_8u = numpy.interp(numpy.arange(0, 256), lut_in, lut_out).astype(numpy.uint8)
    image_contrasted = cv2.LUT(image, lut_8u)
    cv2.imshow('curved with look up table', image_contrasted)
    cv2.waitKey()


def test_photo_2():
    """
    This will server as a test for figuring out good photo quality with python
    """
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = camera.read()
    print('backend name:', camera.getBackendName())
    print(f'resolution: {frame.shape[0]}x{frame.shape[1]}')
    count = 0
    n = 0
    while True :
        count += 1
        ret, frame = camera.read()
        cv2.imshow('camera feed', frame)
        if count > 10 and n <= 5:
            count = 0
            n += 1
            cv2.imwrite('capture' + str(n) + '.png', frame)
            print('took photo!')
            
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    print('done')
                   
            
def test_photo_3():
    """
    This test demonstrates using a live video feed to capture coordinates.
    First, A live video feed is shown.
    On the video feed, you can click, and it will show on the screen
    where the screen was clicked.
    The function returns
    """
    refPoints = []

    def click(event, x, y, flags, param):
        """occurs when left mouse is clicked in opencv"""
        if event == cv2.EVENT_LBUTTONDOWN:
            refPoints.append((x, y))


    camera = cv2.VideoCapture(1)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)



    while True:

        ret, frame = camera.read()
        
        for point in refPoints:
            cv2.circle(frame, (point[0], point[1]), 10, (0, 0, 255), 2)
        
        
        cv2.imshow('camera feed', frame)
        cv2.setMouseCallback('camera feed', click)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    return refPoints
                   



def test_photo_4():
    """
    Draws the contours on the video feed

    -Just draws the outlines on the video feed.
    -I want to see how many I have to pull out before the actual qrcode grid will be captured
    - close it on q
    """
    n = 4  # the biggest n contours
    #vid = cv2.VideoCapture(0)
    vid = find_camera(1)
    while True:
        ret, frame = vid.read()
        cv2.imshow(f'Showing {n} Contours', find_contours(frame, n))  # contours are found and displayed hear
        if cv2.waitKey(1) & 0xFF == ord('q'):  # quit on pressing q
            break
    vid.release()
    cv2.destroyAllWindows()


def test_photo_5():
    """
    See what happens in the contours are ran through a verification function of some sorts
    - perform homography based on contours?
    - every frame, check however many contours
    """
    #vid = cv2.VideoCapture(0)
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
            suspect = extractor.perspective_transform_already_ordered(original, reordered_corners)
            results = calibration_preprocess(suspect, 5, reordered_corners)
            print('calibration preprocess results:', results)
            cv2.imshow('extracted image', suspect)
            cv2.waitKey()


    vid.release()
    cv2.destroyAllWindows()



def try_new_verify():
    """testing the new verification method"""
    vid = find_camera()
    while True:
        ret, frame = vid.read()
        cv2.imshow('Verifying...', frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):  # quit on q
            break
        elif key_pressed == ord('\r'): # return key
            visible_bits = verify(frame, 5)
            print(visible_bits)
            #cv2.waitKey()

    return -1


"""
-----HELPFUL FUNCTIONS------
"""
    
def take_n_save_pic(camera, name):
    ret, frame = camera.read()
    cv2.imwrite(f'debug/{name}.png', frame)

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




def click_for_coordinates(image):
    """copy of click_for_coordinates in obstruction.py"""
    root = tk.Tk()
    original = image
    try:  # converts image to pil image incase it is not
        original = PIL.Image.fromarray(original)
    except:
        pass
    w = tk.Canvas(root, width=original.width + 100, height=original.height)
    w.pack()
    img = PIL.ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")
    text = tk.Text(root, height=6)
    text.insert(tk.END, 'Click to select the coordinates of the corners...\nTop Left:')
    text.pack(side=tk.RIGHT)
    coordinates = []
    def printcoords(event):
        coordinates.append(int(event.x))
        coordinates.append(int(event.y))
        if len(coordinates) >= 8:
            w.quit()
            root.destroy()
    w.bind("<Button 1>", printcoords)
    root.mainloop()
    return coordinates



def find_camera(n=None):
    """ returns the camera capture object itself """
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







def check_this_image(img, n, kernel, sigma, amount):
    """
    This function takes the image, pops it up on the screen so the user can click on the corners. The corner
    locations are used to crop out the qr code grid. The qr code is preprocessed and
    """
    #ensure the kernel is valid
    assert (kernel % 2 > 0) and (kernel > 0), 'Your kernel value (' + str(kernel) + ') sucks.'

    # first, prompt user for the coordinates of the image
    corners = click_for_coordinates(img)

    # perform the homography
    x1, y1, x2, y2, x3, y3, x4, y4 = tuple(corners)
    corners = numpy.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    cropped = extractor.perspective_transform_already_ordered(img, corners)  #color version cropped

    #apply color curve
    cropped = apply_color_curve(cropped)

    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)  #gray scale version, cropped

    # first check the raw results, return and exit if we are good
    raw_results = qrcode2.verify_image_percent(gray, n)
    if raw_results >= 100:
        print('No pre-processing required, returning')
        return raw_results
    else:
        print('raw results:', raw_results)

    # process the image
    sharpened_image = thresh.sharpen(gray, kernel_size=(kernel, kernel), sigma=sigma, amount=amount)

    # check the results of the sharpened image
    sharpened_results = qrcode2.verify_image_percent(sharpened_image, n)
    print('sharpened results:', sharpened_results)


    #iterate over possible values of of sharpen parameters
    kernel_values = [1, 3]
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    amount_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    total = 0
    n_scores = 0
    missing_count = dict()
    for i in range(25):
        missing_count[i] = 0

    for _k, _s, _a in itertools.product(kernel_values, sigma_values, amount_values):
        sharpened_image = thresh.sharpen(gray, kernel_size=(_k, _k), sigma=_s, amount=_a)
        #score = qrcode2.verify_image_percent(sharpened_image, 5)
        ratio, results, missing = qrcode2.verify_image(sharpened_image, 5)
        for code_location in missing:
            missing_count[code_location] += 1

        n_scores += 1
        total += ratio
        #print(f'k:{_k}, s:{_s}, a:{_a}, score:{score}')

    for code_location in range(25):
        new_line = [5, 10, 15, 20]
        if code_location in new_line:
            print('\n-------------------')
        print(missing_count[code_location], end='|')

    # print('average score:', total/n_scores)


def check_this_image_already_cropped(image, n, kernel, sigma, amount):
    # ensure the kernel is valid
    assert (kernel % 2 > 0) and (kernel > 0), 'Your kernel value (' + str(kernel) + ') sucks.'

    # apply color curve
    image = apply_color_curve(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # gray scale version, cropped

    # first check the raw results, return and exit if we are good
    raw_results = qrcode2.verify_image_percent(gray, n)
    if raw_results >= 100:
        print('No pre-processing required, returning')
        return raw_results
    else:
        print('raw results:', raw_results)

    # process the image
    sharpened_image = thresh.sharpen(gray, kernel_size=(kernel, kernel), sigma=sigma, amount=amount)

    # check the results of the sharpened image
    sharpened_results = qrcode2.verify_image_percent(sharpened_image, n)
    print('sharpened results:', sharpened_results)

    # iterate over possible values of of sharpen parameters
    kernel_values = [1, 3]
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    amount_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    total = 0
    n_scores = 0
    missing_count = dict()
    for i in range(25):
        missing_count[i] = 0

    for _k, _s, _a in itertools.product(kernel_values, sigma_values, amount_values):
        sharpened_image = thresh.sharpen(gray, kernel_size=(_k, _k), sigma=_s, amount=_a)
        # score = qrcode2.verify_image_percent(sharpened_image, 5)
        ratio, results, missing = qrcode2.verify_image(sharpened_image, 5)
        for code_location in missing:
            missing_count[code_location] += 1

        n_scores += 1
        total += ratio
        # print(f'k:{_k}, s:{_s}, a:{_a}, score:{score}')

    for code_location in range(25):
        new_line = [5, 10, 15, 20]
        if code_location in new_line:
            print('\n-------------------')
        print(missing_count[code_location], end='|')

    # print('average score:', total/n_scores)



def apply_color_curve(image):

    lut_in = [0, 127, 200]  # right indent?
    lut_out = [0, 52, 255]  # left indent?

    lut_8u = numpy.interp(numpy.arange(0, 256), lut_in, lut_out).astype(numpy.uint8)
    image_contrasted = cv2.LUT(image, lut_8u)
    return image_contrasted




def find_contours(im, n_contours):
    """
    finds the contours in an image
    To be used as a function that returns the image with contours drawn.
    Only draws the biggest contour right now
    """
    #process and find contours
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)  #convert image to grayscale
    ret_val, threshed_image = cv2.threshold(imgray, 125, 255, 0)  # 50:50 thresh
    contours, hierarchy = cv2.findContours(threshed_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) # find the contours
    the_best_around = sorted(contours, key=cv2.contourArea, reverse=True) # sort the images by largest contour first
    cv2.drawContours(im, the_best_around[0:n_contours], -1, (0, 255, 0), 3) # draw the contours on the new image
    return im  # return the image with contours draw on it


def find_best_contour_with_details(im, selected=0):
    """
    finds the contours on the image.
    This function will ALSO return the contour objects themselves, so they can be used to crop thru homography
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
                sharpened_image = thresh.sharpen(gray, kernel_size=(_k, _k), sigma=_s, amount=_a)
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
    #line_with_corners = f'{corners[0][0]},{corners[0][1]};{corners[1][0]},{corners[1][1]};{corners[2][0]},{corners[2][1]};{corners[3][0]},{corners[3][1]}\n'
    line_with_corners = f'{corners[0, 0]},{corners[0, 1]};{corners[1, 0]},{corners[1, 1]};{corners[2, 0]},{corners[2, 1]};{corners[3, 0]},{corners[3, 1]}\n'
    file.write(line_with_corners)
    file.write(f'{kernel},{sigma},{amount}')
    file.close()



def verify(image, n):
    """
    Parse what is located in calibrate.txt to verify the image.
    Image is NOT cropped yet
    Preprocessing levels
        -1 -> error
        0 -> raw image
        1 -> just curve the image
        2 -> curve + sharpen
    """
    file = open('calibrate.txt', 'r')
    preprocessing_level = int(file.readline().strip())  # parse first line to an int
    corners = [(float(corner.split(',')[0]), float(corner.split(',')[1])) for corner in file.readline().strip().split(';')]  # read in the corners
    suspect = extractor.perspective_transform_already_ordered(image, numpy.float32(corners)) # crop out image on homography

    if preprocessing_level == 0: # raw image
        best_results = qrcode2.verify_image(suspect, n)


    elif preprocessing_level == 1: # just curve the image
        suspect = apply_color_curve(suspect)
        best_results = qrcode2.verify_image(suspect, n)


    elif preprocessing_level == 2: # curve and sharpen
        suspect = apply_color_curve(suspect)
        best_results = qrcode2.verify_image(suspect, n)

    else:
        print('ERROR: the preprocessing level set in calibrate.txt is not valid:', preprocessing_level)
        return -1

    bit_string = ''  # 1 if readable, 0 if not readable
    for i in range(n * n):
        bit_string += '0' if i in best_results[2] else '1'  # add the appropriate string to the output

    cv2.imshow('VISIBILITY', draw_missing_components(suspect, bit_string, n))
    return bit_string




def draw_missing_components(image, bits, n):

    img = PIL.Image.fromarray(image) # pil copy of image
    draw = PIL.ImageDraw.Draw(img)  # draw object
    radius = 10  # radius of circles

    # iterate over the subimages
    height, width = image.shape[0], image.shape[1]  # extract width and height of the image
    subheight = height // n
    subwidth = width // n
    x_crossections = []
    y_crossections = []
    for i in range(n):
        x_crossections.append(subwidth * i)
        y_crossections.append(subheight * i)
    qr_code_number = 0
    for y in y_crossections:
        for x in x_crossections:
            x1, y1, x2, y2 = x, y, x + subwidth, y + subheight
            _x, _y = (x1 + x2) // 2, (y1 + y2) // 2
            if bits[qr_code_number] == '1':
                draw.ellipse((_x - radius, _y - radius, _x + radius, _y + radius), fill='red', outline='red')
            else:
                draw.ellipse((_x - radius, _y - radius, _x + radius, _y + radius), fill='blue', outline='blue')
            qr_code_number += 1
    return numpy.array(img)


if __name__ == '__main__':
    """Main"""
    #/home/atl0026/Desktop/Programs/capture3.png
    #check_this_image(img=cv2.imread(input('image path:')), n=5, kernel=3, sigma=1, amount=1)
    #print(qrcode2.verify_image(cv2.imread(r'/Users/aidanlear/Desktop/5-obstructed.png'), 5))
    test_photo_5()
    #try_new_verify()



