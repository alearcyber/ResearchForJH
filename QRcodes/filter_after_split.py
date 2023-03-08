import cv2
import calibrate2
import numpy
import qrcode2
import os

def part_one():
    #read in image of qr code
    im = cv2.imread(r'/Users/aidanlear/Desktop/Screenshot_from_2022-10-14_17-47-19.png')
    original = im.copy()
    image_cont_drawn, contours = calibrate2.find_best_contour_with_details(im)

    selected = 0
    contour = contours[selected % len(contours)]

    peri = cv2.arcLength(contour, True)
    corners = cv2.approxPolyDP(contour, 0.04 * peri, True)  # ORDER: top left, bottom left, bottom right, top right
    corners = [(e[0, 0], e[0, 1]) for e in corners]  # fix the weird formatting
    reordered_corners = numpy.float32(calibrate2.reorder(corners))  # reorder the corners here
    suspect = calibrate2.perspective_transform_already_ordered(original, reordered_corners) # crop it
    cv2.imshow('Contours', image_cont_drawn)
    cv2.imshow('homographfied', suspect)
    cv2.waitKey()



def part_two():
    image = cv2.imread('/Users/aidanlear/Desktop/OUTPUT.png')
    images = qrcode2.grid_out_image(image, 5)
    code_zero = images[0]
    cv2.imshow('Code Zero', code_zero)
    cv2.waitKey()


def part_three():
    print('--ORIGINAL--')
    print(qrcode2.verify_image_pretty(cv2.imread('/Users/aidanlear/Desktop/OUTPUT.png'), 5))
    print('\n--OTSU--')
    image = cv2.imread('/Users/aidanlear/Desktop/OUTPUT-otsu.png')
    #for i in images:
    results = qrcode2.verify_image_pretty(image, 5)
    print(results)



def part_four():
    """
    benchmark new image readability function that combines the results.
    Uses test set of images taken in lab.
    The test set has original images and the ones taken with modified color settings on the display.
    """
    originals_folder = r'/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/TestNewColorsSet/OriginalColors/'
    modified_colors_folder = r'/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/TestNewColorsSet/ModifiedColorSettings/'

    original_paths = next(os.walk(originals_folder), (None, None, []))[2]
    modified_paths = next(os.walk(modified_colors_folder), (None, None, []))[2]

    original_paths = [originals_folder + path for path in original_paths] # originals
    modified_paths = [modified_colors_folder + path for path in modified_paths] # the colors are new


    print('Backend Used:', qrcode2.CURRENT_BACKEND)
    print('\n----OG IMAGES----')
    og_scores = []
    for path in original_paths:
        print(f'--IMAGE: {path}--')
        img = cv2.imread(path)
        #percent = qrcode2.verify_image_with_overlap(img, 5)
        percent, bitstring = qrcode2.verify_image_overlap_fast(img, 5)
        print('percent:', percent)
        qrcode2.pretty_print_bitstring(bitstring, n=5)
        og_scores.append(percent)
    print('Average Score:', sum(og_scores)/len(og_scores))

    print('\n----Modified Images----')
    og_scores = []
    for path in modified_paths:
        print(f'--IMAGE: {path}--')
        img = cv2.imread(path)
        # percent = qrcode2.verify_image_with_overlap(img, 5)
        percent, bitstring = qrcode2.verify_image_overlap_fast(img, 5)
        print('percent:', percent)
        qrcode2.pretty_print_bitstring(bitstring, n=5)
        og_scores.append(percent)
    print('Average Score:', sum(og_scores) / len(og_scores))


def part_five():
    """ benchmarking the different backends """
    originals_folder = r'/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/TestNewColorsSet/OriginalColors/'
    modified_colors_folder = r'/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/TestNewColorsSet/ModifiedColorSettings/'

    original_paths = next(os.walk(originals_folder), (None, None, []))[2]
    modified_paths = next(os.walk(modified_colors_folder), (None, None, []))[2]

    original_paths = [originals_folder + path for path in original_paths] # originals
    modified_paths = [modified_colors_folder + path for path in modified_paths] # the colors are new



    for path in original_paths:
        print(f'----IMAGE: {path}----')
        img = cv2.imread(path)
        qrcode2.verify_image_with_overlap(img, 5)

    for path in modified_paths:
        print(f'----IMAGE: {path}----')
        img = cv2.imread(path)
        qrcode2.verify_image_with_overlap(img, 5)




if __name__ == '__main__':
    #part_four()
    part_five()



"""
- pyzbar
- quirk
- zxing (probably the zebra library)
"""









