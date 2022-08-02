"""
read in the configurations
"""
import cv2

#----configuration variables----

# x1, y1, x2, y2 location of qr code image to be analyzed
MASK = None

#setup the video capture object
video_capture = cv2.VideoCapture(1)

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

