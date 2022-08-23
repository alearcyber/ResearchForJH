"""
@Author Aidan Lear
Driver python file

Options for what can be passed through the pipe right now:
    verify n
    close | exit | done
    calibrate n

    find
        finds the border

    verifymask n
        performs verifications BUT crops with a mask. Mask is passed through python

"""
#config is where all the configurations are stored
import numpy
import config
import qrcode2
import cv2
import thresh
import PIL.Image, PIL.ImageTk
import extractor
import callibration
import os
import tkinter as tk



def capture_image():
    """
        capture image, crop and apply homography based on points set in config.py.
        If the points are NOT set yet, just use the original image. If this is the case, print a warning message
        to indicate that they need to be found
    """

    #use camera OR give it a path
    if input('ATTEMPTING TO TAKE PICTURE... USE CAMERA??(y/n)').startswith('y'):
        # read a single frame from the video stream, convert to grayscale
        ret, frame = config.video_capture.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        img_path = input('Image Path:')
        frame = cv2.imread(img_path, 0)

    #save the image
    config.log_image(frame, 'raw-image-capture')
    config.log('Image taken with camera has been saved.')


    #crop the image
    #check to see if the cropped location even exists
    if config.display_coordinates is None:
        print("WARNING: The coordinates for cropping out the display are missing. You should try to find them by calibrating first.")
        cropped = frame
    else:
        #crop the image based on corners specified in the config file
        cropped = extractor.perspective_transform_already_ordered(frame, config.display_coordinates)


    #log the image that was cropped
    config.log_image(cropped, 'cropped-during-verification')

    #result
    return cropped


def verify(n):
    """
    verify finds areas in the image that are missing
    :param n: it is an nxn grid of images
    :return: returns a list of unreadable zones as integers (as per the reading order would let you know which ones
        they are)

    First, this function will take a picture of the display. It will be cropped to only include the screen based on
    the config file. The cropping is done within the call to capture_image(). Capture image will save a copy of what it
    captured in the debug folder. From there, the image will be preprocessed. pre-processing will consist of converting
    to grayscale and sharpening with the parameters set in config.py The parameters are set to whatever is passed
    through in the config.txt file OR they can be set by sending a calibration request through the pipe. After
    preprocessing, the qr codes will be read. A debug image visually showing what qrcodes could be read or not
    will be saved. The results will be returned. The parse() function will handle sending the results
    """
    #Capture and crop the image
    image = capture_image()

    #preprocess the image (sharpen with the parameters set in config.py)
    preprocessed_image = thresh.sharpen(image,
                                        kernel_size=(config.KERNEL, config.KERNEL),
                                        sigma=config.SIGMA,
                                        amount=config.AMOUNT)



    # verify the image itself
    verification_results = qrcode2.verify_image(img=preprocessed_image, n=n)
    unreadable_zones = verification_results[2]


    # TODO Save a debug image with red x's over bad qr codes and green checkmarks over the good ones
    # TODO CONTINUE MAKING THE DEBUG IMAGE FROM HERE PLEASE
    verification_debug_image = PIL.Image.fromarray(preprocessed_image)
    for box in unreadable_zones:
        pass


    # Return a list of the unreadable zones
    return unreadable_zones



def send_out(message):
    """
    Send a message on the out pipe
    :param message: str -> The message to be sent on the outpipe
    """
    outpipe = open(config.OUT_PIPE_PATH, "w")
    outpipe.write(message)
    outpipe.close()

    #log that what was sent back
    config.log(f'Sent "{message.strip()}" to the FROM AIDAN pipe.')



def parse(data: str):
    """
    Parse data read from the pipe.
    Write the results to the out pipe.
    Options: calibrate, verify, handshake
    """
    tokens = data.split(" ")


    # verification
    if tokens[0].lower() == "verify":
        n = int(tokens[1])
        unreadable_zones = verify(n)

        #make the unreadable zones into a bit array
        output_bit_array = []
        for i in range(n**2):
            if i in unreadable_zones:
                output_bit_array.append(0)
            else:
                output_bit_array.append(1)

        bit_array_string = ""  #holds the results to be sent to jacob
        for i in output_bit_array:
            bit_array_string += str(i)

        print(bit_array_string)



    #handshake
    elif tokens[0] == "1":
        config.log(f'Received Handshake, full string received: {data}')
        send_out('1\n')



    # Calibrate the corners of the points to be extracted
    # return some sort of error or success code
    elif tokens[0] == 'calibrate':
        assert len(tokens) == 2, 'ERROR: invalid tokens read in the pipe for calibration, should just be:\n\tcalibrate n\n' \
                                 'where n is the dimensions of the qr code grid.\n' \
                                 'You passed through this: ' + data

        assert tokens[1].isdigit(), 'ERROR: the second argument for calibrate MUST be a digit, you passed through: ' + tokens[1]
        calibrate(int(tokens[1]))
        #TODO - send back some status message for calibration. Maybe write success or failure (1 or 0)?


    elif tokens[0] == 'find':
        config.log("Input from pipe perceived as a FIND operation. beggining to find the border...")
        #TODO figure out the return value for find(). actually set it up and whatnot
        result = find()


    elif tokens[0] == 'verifymask':
        config.log('Input from pipe perceived as VERIFYMASK. beginning operation.')

        #ensure verifymask was passed correctly
        assert len(tokens) == 2, 'ERROR, incorrect number of tokens for VERIFY MASK. Tokens received: ' + str(tokens)

        # nxn grid of qr codes
        n = int(tokens[1])

        #perform the verification, result should be the bit string to send back to jacob
        result = verify_mask(n)

        #check that the result is valid
        assert len(result) == n*n, f'ERROR: wrong length of bits was sent from verify_mask(), length received:{len(result)}, length expected:{n*n}'
        only1and0 = True #boolean to tell if only 1's and 0's were in the bitlist
        for bit in result:
            only1and0 = only1and0 and (bit in result)
        assert only1and0, f'ERROR: found an invalid character in the bit string returned from verify_mask(), got:' + result

        #Send Results back to Jacob
        send_out(result)



def calibrate(n):
    """
        find the coordinates of the corner.
        Minimize the parameters for sharpening.
        Corner coordinates and the sharpening parameters will be save in memory within config.py
    """

    # ----Debug Messages----
    config.log(f'Beginning calibration sequence with {n}x{n} Grid of qr codes...')

    # ----FIND CORNER COORDINATES----
    frame = config.capture_image('raw')
    config.log_image(frame, 'before_crop')
    cropped, corners = extractor.extract(frame)

    #save the image cropped out for calibration
    config.log_image(cropped, 'calibration-image-cropped')


    #save the corners that were found
    config.display_coordinates = corners


    #status message
    config.log('Successfully set the corner coordinates of the display. Coordinates found: ' + str(corners))



    # ----MINIMIZE SHARPENING PARAMETERS----
    config.log("Beginning Sharpen Parameter Minimization...")
    parameters = callibration.brute_force_sharpen_parameters(cropped, n) # added in a brute force calibration
    # sigma, amount = callibration.calibrate(cropped, n)   # removed the old minimization


    # parameters being None constitutes a failure
    if parameters is None:
        config.log('ERROR: FAILED TO MINIMIZE PARAMETERS FOR SHARPENING')
        send_out('0\n')

    #if not none, we found a minimum
    else:
        #unpack tuple where parameters are stored
        kernel, sigma, amount = parameters
        config.log(f'Completed Minimization. Kernel={kernel} Sigma={sigma}, amount={amount}')
        config.KERNEL = kernel
        config.SIGMA = sigma
        config.AMOUNT = amount
        send_out('1\n')



def find():
    """ THis function will take a picture and attempt to find the border"""
    # capture an image
    image = config.capture_image('raw')

    # debugging information
    config.log_image(image, 'find-precrop')
    config.log('Successfuly captured raw image and saved in the debug location...')

    # attempt with the cropping
    config.log('Finding border...')
    cropped_image, corners = extractor.extract(image)
    config.log('Found border.')
    config.log_image(cropped_image, 'find-postcrop')
    config.log('Saved cropped image...')




def verify_mask(n):
    """
    verification with a mask
    TODO make a good comment here for this
    """
    #log info
    config.log('Starting verifymask...')

    #take and save raw picture
    image = config.capture_image('raw')
    config.log_image(image, 'verifymask-precrop')
    config.log('Took raw image.')


    #prompt the user for the mask
    corners = click_for_coordinates(image)


    #check the the number of coordinates for the corners is correct, should be 8
    assert len(corners) == 8, 'ERROR: Incorrect number of coordinates, number of coordinates received: ' + str(len(corners))


    # unpack corners; 1=topleft, 2=topright, 3=bottomright, 4=bottomleft
    x1, y1, x2, y2, x3, y3, x4, y4 = tuple(corners)
    assert x2 > x1, 'ERROR: the topleft and topright corners are potentially out of place'
    assert x3 > x4, 'ERROR: the bottomleft and bottomright corners are potentially out of place'
    assert y4 > y1, 'ERROR: the topleft and bottomleft corners are potentially out of place'
    assert y3 > y2, 'ERROR: the topright and bottomright corners are potentially out of place'
    assert x3 > x1, 'ERROR: the topleft and bottomright corners are potentially out of place'
    assert y3 > y1, 'ERROR: the topleft and bottomright corners are potentially out of place'


    #log what parameters were passed, let user know everything is good
    config.log(f'Working with a {n}x{n} qr code grid...')
    config.log(f'CORNERS: ({x1}, {y1}), ({x2}, {y2}), ({x3},{y3}), ({x4}, {y4})')



    #homography/perspective transform based on coordinates
    #ORDER IMPORTANT: topleft, topright, bottomright, bottomleft
    corners = numpy.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    cropped = extractor.perspective_transform_already_ordered(image, corners)

    #conver cropped to grayscale

    config.log_image(cropped, 'verifymask-postcrop')
    config.log('Successfully took picture and cropped. Beginning verification...')

    #find percentage for the raw then the cropped
    raw_results = qrcode2.verify_image(cropped, n)
    cropped = thresh.sharpen(cropped, kernel_size=(config.KERNEL, config.KERNEL),
                                    sigma=config.SIGMA,
                                    amount=config.AMOUNT)
    processed_results = qrcode2.verify_image(cropped, n)


    #Determine if the processed image or raw image was easier to read based on the results from verify...
    if raw_results[0] > processed_results[0]:
        config.log('Raw Image verified higher than raw results')
        best_results = raw_results
    else:
        config.log('Processed image verified higher than raw.')
        best_results = processed_results

    #let the user know what was found
    config.log(f'Completed Verification. {round(best_results[0] * 100)}% of the qrcodes were identified.')


    # construct and return the bit string, 0 if the qrcode was obstructed, 1 if qr code was read
    bit_string = ''
    for i in range(n*n):
        bit_string += '0' if i in best_results[2] else '1' # add the appropriate string to the output


    #make a debug image for if the image verified correctly
    config.log_verification(cropped, bit_string, n)

    #return the results back to the parser
    return bit_string







def click_for_coordinates(image):
    """
    Opens a window with the given image in in and allows the user to click 4 times. The function then
    returns the coordinates of the 4 points select in the form of [x1, y1, x2, y2, x3, y3, x4, y4] where
    1=topleft, 2=topright, 3=bottomright, 4=bottomleft

    """
    config.log('Prompting User for corners, opening window for corner selection...')

    root = tk.Tk()

    original = image
    try:  # converts image to pil image incase it is not
        original = PIL.Image.fromarray(original)
    except:
        pass

    # setting up a tkinter canvas
    w = tk.Canvas(root, width=original.width + 100, height=original.height)
    w.pack()

    img = PIL.ImageTk.PhotoImage(original)
    w.create_image(0, 0, image=img, anchor="nw")

    # add the text
    text = tk.Text(root, height=6)
    text.insert(tk.END, 'Click to select the coordinates of the corners...\nTop Left:')
    text.pack(side=tk.RIGHT)

    next_corner_text = ['', 'Bottom Left: ', 'Bottom Right: ', 'Top Right: ']
    coordinates = []
    def printcoords(event):

        try:
            text_to_add = f'({event.x}, {event.y})\n{next_corner_text.pop()}'
            text.insert(tk.END, text_to_add)
            coordinates.append(int(event.x))
            coordinates.append(int(event.y))
        except:
            # add done message to let the user know they are done
            root.destroy()

        try:
            if len(coordinates) == 8:
                text.insert(tk.END, 'Four coordinates set.. click again to close this window...')
        except:
            pass

    w.bind("<Button 1>", printcoords)
    root.mainloop()

    #log and return
    config.log(f'Finished selecting coordinates: '
               f'({coordinates[0]}, {coordinates[1]}), ({coordinates[2]}, {coordinates[3]}), ({coordinates[4]}, {coordinates[5]}), ({coordinates[6]}, {coordinates[7]}) ')
    return coordinates





def main():
    """
    This is the function that waits for input on the pipe, parses, and writes back to pipe
    """
    FIFO = config.PIPE_PATH
    terminating = False
    print("Opening Pipe...")
    # print a debug message for locations of the pipes
    config.log("Connecting to FROM JACOB pipe. Located at: " + os.path.abspath(FIFO))
    config.log('FROM AIDAN pipe is located at: ' + os.path.abspath(config.OUT_PIPE_PATH))


    while True:
        with open(FIFO) as fifo:
            while True:
                print('awaiting input from pipe')
                data = fifo.read().strip().strip('\x00')

                # i think this resets the writer as to not block
                if len(data) == 0:
                    break

                #Check for termination code
                if data.startswith(('done', 'exit', 'close')):
                    terminating = True
                    print("Closing Pipe...")

                #Log what was received on the pipe
                config.log(f'Input from pipe received: {data}')


                #send data to the parser
                parse(data)


            if terminating:
                fifo.close()
                break
        if terminating:
            break



if __name__ == '__main__':
    #image = cv2.imread(input('image path:'))
    #click_for_coordinates(image)

    main()
