"""
read in the configurations
"""
import cv2
from datetime import datetime
import os
import PIL.Image

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
    img = PIL.Image.fromarray(image)
    img.save(filepath, 'png')


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


#log for standalone mode
if STANDALONE_MODE:
    log('Program running in Standalone Mode')
