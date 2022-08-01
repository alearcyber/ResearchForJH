import cv2
import qrcode
import PIL.Image
import numpy

detector = cv2.QRCodeDetector()  # cv2 qr code reader object

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



def read_qr():
    """
    read the qr codes with detect and decode multi
    """
    n = 4
    #img = cv2.imread("/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qr-images/qr4x4picThresh.png")
    img = cv2.imread("/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qr-images/qr5x5pic.png")
    results = detector.detectAndDecodeMulti(img)[1]
    results = sorted([int(result) for result in results])
    print(results)


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
        result = detector.detectAndDecode(imgs[i])[0]
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







def old_stuff():
    """this was in main when i deleted it to make qr codes for jacob"""
    n = 5
    #i = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qrcodepics/qr3.png")
    i = cv2.imread(r"/Users/aidanlear/Desktop/qrcodes png/qr5thresh.png", 0)
    #i = cv2.medianBlur(i, 5)
    i = cv2.adaptiveThreshold(i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    show_ndarray(i)
    #nothin, i = cv2.threshold(i, 125, 255, cv2.THRESH_BINARY)  # with threshholding
    print(verify_image(i, n))
    images = grid_out_image(i, n)
    show_ndarray(images[20])
    #images = grid_out_image(i, n)



def get_codes_for_jacob():
    """get the qr code raw images for jacob"""







if __name__ == '__main__':
    """driver"""











