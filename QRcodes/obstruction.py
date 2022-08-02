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




def capture_image():
    """capture image and crop to specifications outlined in config.txt"""
    #read a single frame from the video stream, convert to grayscale
    ret, frame = config.video_capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #crop the image based on the mask specified in the config file
    cropped: numpy.ndarray = qrcode2.crop_image(frame, config.MASK)

    #save the image to the debug folder
    time = datetime.now().strftime("%H-%M-%S")
    filename = config.DEBUG_FOLDER + 'capture-at-' + time + '.png'
    qrcode2.save_ndarray(cropped, filename)



    #result
    return cropped


def verify(n):
    """
    verify finds areas in the image that are missing
    :param n: it is an nxn grid of images
    :return: returns a list of unreadable zones as integers (as per the reading order would let you know which ones
        they are)

    First, this function will take a picture of the display. It will be cropped to only include the screen based on.
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
    for box in unreadable_zones:
        verification_debug_image = PIL.Image.fromarray(preprocessed_image)

    # Return a list of the unreadable zones
    return unreadable_zones



def parse(data: str):
    """
    Parse data read from the pipe.
    Write the results to the out pipe.
    Options: calibrate, verify, handshake
    """
    tokens = data.split(" ")
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
