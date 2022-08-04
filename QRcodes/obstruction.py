"""
@Author Aidan Lear
Driver python file

Options for what can be passed through the pipe right now:
    verify n
    close | exit | done
"""
#config is where all the configurations are stored
import numpy
import config
import qrcode2
import cv2
from datetime import datetime
import thresh
import PIL.Image
import extractor
import callibration




def capture_image():
    """
        capture image, crop and apply homography based on points set in config.py.
        If the points are NOT set yet, just use the original image. If this is the case, print a warning message
        to indicate that they need to be found
    """
    #read a single frame from the video stream, convert to grayscale
    ret, frame = config.video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


    #save the raw capture to the debug folder for debugging
    time = datetime.now().strftime("%H-%M-%S")
    filename = config.DEBUG_FOLDER + 'raw-capture-at-' + time + '.png'
    qrcode2.save_ndarray(frame, filename)



    #crop the image
    #check to see if the cropped location even exists
    if config.display_coordinates is None:
        print("WARNING: The coordinates for cropping out the display are missing. You should try to find them by calibrating first.")
        cropped = frame
    else:
        #crop the image based on corners specified in the config file
        cropped = extractor.perspective_transform(frame, config.display_coordinates)


    #save the cropped image to the debug folder
    filename = config.DEBUG_FOLDER + 'cropped-at-' + time + '.png'
    qrcode2.save_ndarray(cropped, filename)

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
        print("recieved handshake from jacob...")
        outpipe = open(config.OUT_PIPE_PATH, "w")
        outpipe.write('1\n')
        outpipe.close()



    # Calibrate the corners of the points to be extracted
    # return some sort of error or success code
    elif tokens[0] == 'calibrate':
        assert len(tokens) == 2, 'ERROR: invalid tokens read in the pipe for calibration, should just be:\n\tcalibrate n\n' \
                                 'where n is the dimensions of the qr code grid.\n' \
                                 'You passed through this: ' + data

        assert tokens[1].isdigit(), 'ERROR: the second argument for calibrate MUST be a digit, you passed through: ' + tokens[1]
        calibrate(int(tokens[1]))
        #TODO - send back some status message for calibration. Maybe write success or failure (1 or 0)?





def calibrate(n):
    """
        find the coordinates of the corner.
        Minimize the parameters for sharpening.
        Corner coordinates and the sharpening parameters will be save in memory within config.py
    """
    # TODO - Finish writing the calibrator


    # ----FIND CORNER COORDINATES----
    ret, frame = config.video_capture.read()
    cropped, corners = extractor.extract(frame)

    #debug save the raw and the cropped
    time = datetime.now().strftime("%H-%M-%S")
    filename = config.DEBUG_FOLDER + 'raw-capture-for-calibration-at-' + time + '.png'
    qrcode2.save_ndarray(frame, filename)
    filename = config.DEBUG_FOLDER + 'cropped-during-calibration-at-' + time + '.png'
    qrcode2.save_ndarray(cropped, filename)

    #save the corners that were found
    config.display_coordinates = corners

    #status message
    print('Successfully set the corner coordinates of the display. Coordinates found:', corners)




    # ----MINIMIZE SHARPENING PARAMETERS----
    print('Finding best pre-processing parameters...')
    sigma, amount = callibration.calibrate(cropped, n)
    print(f'Done! Sigma={sigma}, amount={amount}')
    config.SIGMA = sigma
    config.AMOUNT = amount







def main():
    """
    This is the function that waits for input on the pipe, parses, and writes back to pipe
    """
    FIFO = config.PIPE_PATH
    terminating = False
    print("Opening Pipe...")
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
                print(f'Input from pipe:{data}')


                #send data to the parser
                parse(data)


            if terminating:
                fifo.close()
                break
        if terminating:
            break



if __name__ == '__main__':
    #test_image = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)
    #find_missing_locations(test_image, 5)
    #capture_image()
    main()
