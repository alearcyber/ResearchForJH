import cv2
import qrcode
import PIL.Image
import numpy
import math
from pyzbar.pyzbar import decode
import zxing

#----CONSTANTS----
#backends
OPENCV = 1
PYZBAR = 2
ZXING = 3
QUIRC = 4
CURRENT_BACKEND = PYZBAR


def show_ndarray(array: numpy.ndarray):
    """shows a numpy array"""
    img = PIL.Image.fromarray(array) # convert to PIL.Image
    img.show() # show the image


def crop_image(image, mask):
    """ mask is (x1, y1, x2, y2)"""
    x1, y1, x2, y2 = mask  # unpack mask tuple
    image = image[y1:y2, x1:x2]
    return image


def save_ndarray(array: numpy.ndarray, out: str):
    """ save an nd array as a pil image where the full path of the image is out """
    img = PIL.Image.fromarray(array)
    img.save(out, 'png')




def creategrid(n, error_correction=qrcode.constants.ERROR_CORRECT_H) -> PIL.Image:
    """
    creates an n by n image of qr codes
    Default error correction is high
    """
    length = 115
    images = []
    for i in range(n*n):
        code = str(i)
        qr = qrcode.QRCode(error_correction=error_correction, border=1, box_size=5)
        qr.add_data(code)
        qr.make()
        img: PIL.Image.Image = qr.make_image(fill_color='black', back_color="white").convert('RGB')
        images.append(img)
    images.reverse()
    new_image = PIL.Image.new('RGB', (n * length, n * length), (250, 250, 250))

    for y in range(n):
        for x in range(n):
            img = images.pop()
            new_image.paste(img, (length * x, length * y))

    #new_image.show()
    return new_image




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






def read_qrcode(img, backend=CURRENT_BACKEND):
    """
    return a string representation of what was scanned from a single qr code.
    This function gives the ability to choose which backend to use, default is whatever selected, seleced starts as opencv
    Current implementet backends: opencv, pyzbar, zxing
    """

    assert backend in [OPENCV, PYZBAR, QUIRC, ZXING], 'Did not select a valid backend for reading qr codes'


    #opencv
    if backend == OPENCV:
        detector = cv2.QRCodeDetector()  # cv2 qr code reader object
        result = detector.detectAndDecode(img)[0]
        del detector
        return result

    # pyzbar
    elif backend == PYZBAR:
        results = decode(img)
        if len(results) == 0:
            return ''
        result = results[0].data.decode('utf-8')
        return result


    #zxing
    elif backend == ZXING:
        reader = zxing.BarCodeReader()
        cv2.imwrite('TEMP.png', img)
        result = reader.decode('TEMP.png').raw
        if result is None:
            return ''
        else:
            return result


    #Quirc
    elif backend == QUIRC:  # TODO implement quirc
        assert False, 'WARNING: Quirc backend has NOT been implemented yet'








def grid_out_image(image, n):
    """split the image into n x n sub-images, return in list, ordered like reading order"""
    height, width = image.shape[0], image.shape[1] # extract width and height of the image
    subheight = height//n
    subwidth = width//n

    subimages = []

    x_crossections = []
    y_crossections = []

    for i in range(n):
        x_crossections.append(subwidth * i)
        y_crossections.append(subheight * i)

    for y in y_crossections:
        for x in x_crossections:
            x1, y1, x2, y2 = x, y, x + subwidth, y + subheight

            if x2 > width:
                x2 = width
            if y2 > height:
                y2 = height

            mask = x1, y1, x2, y2
            cropped_section = crop_image(image, mask)
            subimages.append(cropped_section)

    return subimages



def verify_image(img, n, show_unreadable_codes=False):
    """
    chop up the image into subimages and see what qr codes can be read or not.
    Returns ratio, results, missing
    """
    imgs = grid_out_image(img, n)

    results = []

    for i in range(n*n):
        result = read_qrcode(imgs[i])
        results.append(result)

    missing = []
    for i in range(n*n):
        if not str(i) in results:
            missing.append(i)

    ratio = (len(results) - len(missing)) / len(results)

    # Area where the unreadable codes are shown
    if show_unreadable_codes:
        for i in missing:
            show_ndarray(imgs[i])

    return ratio, results, missing


def verify_image_percent(img, n):
    """
    wrapper for image verification that just gives the percent
    """
    percent = verify_image(img, n)[0]
    return round(percent * 100)


def verify_image_pretty(img, n):
    """
    verify the image and return a pretty string as the result
    """
    ratio, results, missing = verify_image(img, n)
    out = ''
    counter = 0
    for i in range(n**2):
        if i in missing:
            out += 'X '
        else:
            out += 'O '
        counter += 1
        if counter >= 5:
            out += '\n'
            counter = 0
    return out



def apply_color_curve(image):
    """
    Apply the color curve like in GIMP
    """
    lut_in = [0, 127, 200]  # right indent?
    lut_out = [0, 52, 255]  # left indent?
    lut_8u = numpy.interp(numpy.arange(0, 256), lut_in, lut_out).astype(numpy.uint8)
    image_contrasted = cv2.LUT(image, lut_8u)
    return image_contrasted




def verify_image_binary(img, n):
    """
    overload of verify image.
    returns the results in binary
    """
    ratio, results, missing = verify_image(img, n)
    out = ''
    for i in range(n**2):
        if i in missing:
            out += '0'
        else:
            out += '1'
    return out, round(ratio*100)


def bitwise_or(bytes1: str, bytes2: str):
    """
    bitwise or on a string of bits for adding up results
    """
    # Do the operation
    result = ''
    for i in range(len(bytes1)):
        if (bytes1[i] == '1') or (bytes2[i] == '1'):
            result += '1'
        else:
            result += '0'
    return result

def pretty_print_bitstring(bitstring, n=None):
    """
    given an image verification bitstring, print out a pretty version of it.
    """
    if n is None:
        n = round(math.sqrt(len(bitstring)))
    out = ''
    for i in range(len(bitstring)):
        if bitstring[i] == '0':
            out += 'X '
        else:
            out += 'O '

        if ((i+1) % n) == 0:
            out += '\n'
    print(out)


def verify_image_with_overlap(img, n, show_mixed=False):
    """
    verify the image by combining results:
        - Raw
        - Otsu's Threshold
        - Color curved
        - unsharp mask

    Results are returned as a pretty print for now
    """
    area = n**2   # area of the grid
    results_list = []

    #raw image
    print('--Raw Image--')
    raw_bitstring, raw_percent = verify_image_binary(img, n)
    out = ''
    for i in range(area):
        if raw_bitstring[i] == '0':
            out += 'X '
        else:
            out += 'O '

        if ((i+1) % n) == 0:
            out += '\n'
    print(out)
    results_list.append(raw_bitstring)



    #otsus threshold
    print('--Otsu\'s Threshold--')
    ret3, otsu_image = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    otsu_bitstring, otsu_percent = verify_image_binary(otsu_image, n)
    out = ''
    for i in range(area):
        if otsu_bitstring[i] == '0':
            out += 'X '
        else:
            out += 'O '

        if ((i + 1) % n) == 0:
            out += '\n'
    print(out)
    results_list.append(otsu_bitstring)


    #Color curve
    print('--Color Curve--')
    color_curved_image = apply_color_curve(img)
    curved_bitstring, curved_percent = verify_image_binary(color_curved_image, n)
    out = ''
    for i in range(area):
        if curved_bitstring[i] == '0':
            out += 'X '
        else:
            out += 'O '

        if ((i + 1) % n) == 0:
            out += '\n'
    print(out)
    results_list.append(curved_bitstring)


    #Unsharp Mask
    print('--Unsharp Mask--')
    sharpened_image = sharpen(img)  # use the defaults for now
    sharpened_bitstring, sharpened_percent = verify_image_binary(sharpened_image, n)
    out = ''
    for i in range(area):
        if sharpened_bitstring[i] == '0':
            out += 'X '
        else:
            out += 'O '

        if ((i + 1) % n) == 0:
            out += '\n'
    print(out)
    results_list.append(sharpened_bitstring)


    # show readability arrays
    if show_mixed:
        print('--Readability--  (O=always; M=at least once; X=never)')
    else:
        print('--Readability--')

    expected_reads = len(results_list)
    out = ''
    total_readable_codes = 0
    for i in range(area):
        read_count = 0
        for bit_list in results_list:  # count the number of times it is read
            if bit_list[i] == '1':
                read_count += 1

        if read_count >= expected_reads:
            out += 'O '
            total_readable_codes += 1
        elif read_count > 0:
            out += ('M ' if show_mixed else 'O ')
            total_readable_codes += 1
        else:
            out += 'X '

        if ((i + 1) % n) == 0:
            out += '\n'
    print(out)

    #calculate the return results as a percentage
    return round(100 * total_readable_codes/area)




def verify_image_overlap_fast(img, n, debug=False):
    """
    fast version of verify_image_with_overlap(). Is less Verbose.
    Returns a tuple, (percent, bitstring)
    """
    area = n**2

    # raw
    bitstring, raw_percent = verify_image_binary(img, n)
    if raw_percent == 100:
        if debug:
            print('Only needed raw image')
        return 100, bitstring


    # otsu
    ret3, otsu_image = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    otsu_bitstring, otsu_percent = verify_image_binary(otsu_image, n)
    if otsu_percent == 100:
        return 100, otsu_bitstring
    bitstring = bitwise_or(otsu_bitstring, bitstring)
    if not bitstring.__contains__('0'):
        if debug:
            print('Stopped after Otsus')
        return 100, bitstring


    # color curve
    color_curved_image = apply_color_curve(img)
    curved_bitstring, curved_percent = verify_image_binary(color_curved_image, n)
    bitstring = bitwise_or(curved_bitstring, bitstring)
    if not bitstring.__contains__('0'):
        if debug:
            print('stopped after color curve')
        return 100, bitstring

    # unsharp mask
    sharpened_image = sharpen(img)  # use the defaults for now
    sharpened_bitstring, sharpened_percent = verify_image_binary(sharpened_image, n)
    bitstring = bitwise_or(sharpened_bitstring, bitstring)
    if not bitstring.__contains__('0'):
        if debug:
            print('stopped after Unsharp Mask')
        return 100, bitstring


    #if made it to here, some qr codes(s) could not be read
    percent = round(100 * bitstring.count('1')/area)
    if debug:
        print(f'Verification was {percent} after combining all the preprocessing methods')
    return percent, bitstring




def test_thresholding():
    """messing around with thresholding an image"""
    tresholds = [  # list[(name, thresholdtype)]
        ('adaptive mean', cv2.ADAPTIVE_THRESH_MEAN_C),
        ('adaptive gaussian', cv2.ADAPTIVE_THRESH_GAUSSIAN_C),
        ('binary', cv2.THRESH_BINARY),
    ]

    img = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qr-images/qr4x4pic.png")
    ratio, results, missing = verify_image(img, 4)
    print('pre-thresh:', ratio)

    ret, thresh = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)
    ratio, results, missing = verify_image(thresh, 4)
    print('post-thresh:', ratio)

    show_ndarray(thresh)






def get_codes_for_jacob():
    """get the qr code raw images for jacob"""
    for i in range(3, 11):
        creategrid(i, qrcode.constants.ERROR_CORRECT_L).save(f'/Users/aidanlear/Desktop/new test images/gridL{i}.png', 'png')
        creategrid(i, qrcode.constants.ERROR_CORRECT_M).save(f'/Users/aidanlear/Desktop/new test images/gridM{i}.png', 'png')
        creategrid(i, qrcode.constants.ERROR_CORRECT_Q).save(f'/Users/aidanlear/Desktop/new test images/gridQ{i}.png', 'png')
        creategrid(i, qrcode.constants.ERROR_CORRECT_H).save(f'/Users/aidanlear/Desktop/new test images/gridH{i}.png', 'png')






def test_different_backends():
    test_image = cv2.imread(r'/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qr-images/qrcode1.png')
    bad_image = cv2.imread('/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qr-images/cracked_screen.png')
    expected_result = 'one'


    #pyzbar tests
    n_test = 1  # test number
    pyzbar_result = read_qrcode(test_image, backend=PYZBAR)
    test = pyzbar_result == expected_result
    print(f'passed test {n_test}' if test else f'FAILED test {n_test}')


    n_test = 2
    pyzbar_result2 = read_qrcode(bad_image, backend=PYZBAR)
    test = pyzbar_result2 == ''
    print(f'passed test {n_test}' if test else f'FAILED test {n_test}')


    #zxing tests
    n_test = 3
    zxing_result1 = read_qrcode(test_image, backend=ZXING)
    test = zxing_result1 == expected_result
    print(f'passed test {n_test}' if test else f'FAILED test {n_test}')


    n_test = 4
    test = read_qrcode(bad_image, backend=ZXING) == ''
    print(f'passed test {n_test}' if test else f'FAILED test {n_test}')


    #checking to see if changing the backends works
    n_test = 5





if __name__ == '__main__':
    """driver"""
    test_different_backends()











