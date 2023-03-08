"""
read in the configurations
"""
import cv2
from datetime import datetime
import os
import PIL.Image, PIL.ImageDraw

#----configuration variables----

# x1, y1, x2, y2 location of qr code image to be analyzed
MASK = None

#setup the video capture object
video_capture = cv2.VideoCapture(0)

#path where debug files are stored
DEBUG_FOLDER = None

#path to the input pipe
PIPE_PATH = None

#to jacob pipe path
OUT_PIPE_PATH = None

#kernel, sigma, and amount
KERNEL = 3
SIGMA = 1.0
AMOUNT = 1.0


#DISPLAY LOCATION, numpy array with the ordered coordinates
display_coordinates = None


#Standalone mode: if this is set to True, it means the program is intended to be ran WITHOUT Jacob's
#portion of the code
STANDALONE_MODE = True


# what time does the program start running, for logging purposes
START_TIME = datetime.now().strftime("%H-%M-%S")


# what is the default camera number. Set to None if there isn't one
DEFAULT_CAM = None


#read configuration file
file = open(r'config.txt', 'r')

for line in file:
    if line.startswith('#'):
        continue
    if '=' not in line:
        continue
    try:
        tokens = line.strip().split('=')
        configuration_name = tokens[0].lower()
        configuration_value = tokens[1]
    except:
        continue

    #----parse out configurations----

    #mask
    if configuration_name == 'mask':
        MASK = tuple(int(element) for element in configuration_value.split(','))


    #debug path
    elif configuration_name == 'debugpath':
        DEBUG_FOLDER = configuration_value

    #pipe location
    elif configuration_name == 'pipe':
        PIPE_PATH = configuration_value

    #output pipe
    elif configuration_name == 'pipetojacob':
        OUT_PIPE_PATH = configuration_value

    #kernel, sigma, and amount
    elif configuration_name == 'kernel':
        KERNEL = int(configuration_value)
    elif configuration_name == 'sigma':
        SIGMA = float(configuration_value)
    elif configuration_name == 'amount':
        AMOUNT = float(configuration_value)


file.close()



def log(message):
    """
    log a message.
    ALso Write to a log file
    """
    print('[LOG]', message)


def log_image(image, name):
    """
    :param image: ndarray -> image to be saved
    :param name: str -> name of the image WITHOUT path or .png at the end
    """
    filename = f'{START_TIME}-{name}.png'
    filepath = os.path.join(DEBUG_FOLDER, filename)
    try:
        img = PIL.Image.fromarray(image)
        img.save(filepath, 'png')
    except:
        image.save(filepath, 'png')



def log_verification(image, bitstring, n):
    """
    saves an image of what was recognized
    TODO add a good comment here
    :param image:
    :param bitstring:
    :return:
    """
    # make a PIL image
    img = PIL.Image.fromarray(image)

    #grab drawing object for the PIL image
    draw = PIL.ImageDraw.Draw(img)

    #radius of the circles to be drawn, in pixels.
    radius = 10

    #iterate over the subimages
    height, width = image.shape[0], image.shape[1] # extract width and height of the image
    subheight = height//n
    subwidth = width//n
    x_crossections = []
    y_crossections = []
    for i in range(n):
        x_crossections.append(subwidth * i)
        y_crossections.append(subheight * i)

    qr_code_number = 0
    for y in y_crossections:
        for x in x_crossections:

            #square where the qr code is
            x1, y1, x2, y2 = x, y, x + subwidth, y + subheight

            #get the coordinates to draw the circle
            _x, _y = (x1 + x2)//2, (y1 + y2)//2

            #draw the circle
            color = 'blue'
            if bitstring[qr_code_number] == '1':
                draw.ellipse((_x - radius, _y - radius, _x + radius, _y + radius), fill=color, outline=color)

            #iterate the qrcode number so we know which one we are on. It's basically an iterator
            qr_code_number += 1


    #call the original log image to save the image.
    log_image(img, 'verify-result')



def capture_image(mode='gray'):
    """
    :param mode: 'raw' or 'gray' for just the raw image or the grayscale copy
    :return: the image as an ndarray
    """
    if input('ATTEMPTING TO TAKE PICTURE... USE CAMERA??(y/n)').startswith('y'):
        # read a single frame from the video stream, convert to grayscale
        ret, frame = video_capture.read()
        if mode == 'gray':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        img_path = input('Image Path:')
        frame = cv2.imread(img_path)

    return frame


print(START_TIME)
#log for standalone mode
if STANDALONE_MODE:
    log('Program running in Standalone Mode')
