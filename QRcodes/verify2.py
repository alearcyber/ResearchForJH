"""
Performs verification of the qrcodes
"""
import os
import cv2
import calibrate2 as calibrate
import numpy
import qrcode2
import PIL.Image, PIL.ImageDraw

"""
---------------CONSTANTS--------------
"""
#pipe path
PIPE_PATH = r'fromJacob'

#pipe path for writing to jacob
OUT_PIPE_PATH = r'fromAidan'

#default camera
DEFAULT_CAM = 0

#delay to allow the camera to focus
CAM_TIMER = 50


"""
----------------------------------------UTILITY FUNCTIONS----------------------------------------
"""
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
    #parse calibrate.txt
    file = open('calibrate.txt', 'r')
    preprocessing_level = int(file.readline().strip())  # parse first line to an int
    corners = [(float(corner.split(',')[0]), float(corner.split(',')[1])) for corner in file.readline().strip().split(';')]  # read in the corners
    suspect = calibrate.perspective_transform_already_ordered(image, numpy.float32(corners)) # crop out image on homography

    #decide on what to do
    if preprocessing_level == 0: # raw image
        best_results = qrcode2.verify_image(suspect, n)


    elif preprocessing_level == 1: # just curve the image
        suspect = calibrate.apply_color_curve(suspect)
        best_results = qrcode2.verify_image(suspect, n)


    elif preprocessing_level == 2: # curve and sharpen
        suspect = calibrate.apply_color_curve(suspect)
        best_results = qrcode2.verify_image(suspect, n)

    else:
        print('ERROR: the preprocessing level set in calibrate.txt is not valid:', preprocessing_level)
        return -1

    bit_string = ''  # 1 if readable, 0 if not readable
    for i in range(n * n):
        bit_string += '0' if i in best_results[2] else '1'  # add the appropriate string to the output

    cv2.imshow('VISIBILITY', draw_missing_components(suspect, bit_string, n))
    return bit_string



def send_out(message):
    """
    Send a message on the out pipe
    :param message: str -> The message to be sent on the outpipe
    """
    outpipe = open(OUT_PIPE_PATH, "w")
    outpipe.write(message)
    outpipe.close()

    #log that what was sent back
    print(f'Sent "{message.strip()}" to the FROM AIDAN pipe.')



"""
----------------------------------------DRIVER FUNCTIONS----------------------------------------
"""

def try_new_verify(n):
    """Driver function"""
    vid = calibrate.find_camera(DEFAULT_CAM)
    timer = CAM_TIMER
    while True:
        timer -= 1
        ret, frame = vid.read()
        cv2.imshow('Verifying...', frame)
        if timer <= 0:
            #verify
            visible_bits = verify(frame, n)

            print(visible_bits)

            #send bits back to jacob
            send_out(str(visible_bits))


        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):  # quit on q
            break

    return -1


def try_new_verify_timer():
    """
    Same as try_new_verify() with difference in how pictures are taken.
    The program waits for jacob to prompt it for an image.
    When Jacob prompts it, the window is opened, and a timer is set, the program waits and THEN takes the picture.
    """
    terminating = False
    print("Opening Pipe...")
    # print a debug message for locations of the pipes
    print("Connecting to FROM JACOB pipe. Located at: " + os.path.abspath(PIPE_PATH))
    print('FROM AIDAN pipe is located at: ' + os.path.abspath(OUT_PIPE_PATH))


    while True:
        with open(PIPE_PATH) as fifo:
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
                print(f'Input from pipe received: {data}')


                #process a "verify n" from Jacob
                tokens = data.split(" ")
                if tokens[0].lower() == "verify":
                    n = int(tokens[1])
                    try_new_verify(n)



            if terminating:
                fifo.close()
                break
        if terminating:
            break




if __name__ == '__main__':
    try_new_verify_timer()
