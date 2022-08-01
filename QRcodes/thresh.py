"""
This file will serve as holding the thresholding functionality

It will also do all the preprocessing


"""


import cv2
import numpy
import PIL.Image
import qrcode2


#Constants

#enum for the thresholding types i want to cover
THRESH_ADAPTIVE_MEAN = 0
THRESH_ADAPTIVE_GAUSSIAN = 1
THRESH_OTSU = 2
THRESH_OTSU_2D = 3

#tresholds as a list
thresh_types = [
    THRESH_ADAPTIVE_MEAN,
    THRESH_ADAPTIVE_GAUSSIAN,
    THRESH_OTSU,
    THRESH_OTSU_2D,
]

def show_ndarray(array: numpy.ndarray):
    """shows a numpy array"""
    img = PIL.Image.fromarray(array) # convert to PIL.Image
    img.show() # show the image



def adaptive_mean(img: numpy.ndarray, block_size, c):
    """
    Perform an adaptive mean threshold on the given image
    :param img: numpy.ndarray - the image to be processed
    :return: numpy.nd array - processed version of the image
    """
    #TODO insert a test to ensure the image is in grayscale



    # invoke the processing function from opencv
    # NOTE ORIGINAL block size was 11
    result: numpy.ndarray = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, c)
    return result



def adaptive_gaussian(img: numpy.ndarray, block_size, c):
    """
    Perform an adaptive gaussian threshold on the given image
    :param img: numpy.ndarray - the image to be processed
    :return: numpy.nd array - processed version of the image
    """
    #TODO insert a test to ensure the image is in grayscale
    result: numpy.ndarray = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    return result



def otsu(img: numpy.ndarray) -> numpy.ndarray:
    """
    ---Otus's Binarization---
    This is an implementation of Otsu's Binarization for image thresholding.
    This automatically determines the value by which to threshold.
    The way it works is by minimizing the variance between the two peaks made by the thresholding.
    Returns a processed version of the numpy array.

    ---params---
    :param img: numpy.ndarray - The image to be processed as a numpy array
    :return: numpy.ndarray - the resulting image

    ---NOTES---
    This version DOES NOT apply any gaussian blurring. I will implement this outside the function.
    """

    blur = img
    # find normalized_histogram, and its cumulative distribution function
    hist = cv2.calcHist([blur], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / hist.sum()
    Q = hist_norm.cumsum()
    bins = numpy.arange(256)
    fn_min = numpy.inf
    thresh = -1
    for i in range(1, 256):
        p1, p2 = numpy.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 < 1.e-6 or q2 < 1.e-6:
            continue
        b1, b2 = numpy.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = numpy.sum(p1 * b1) / q1, numpy.sum(p2 * b2) / q2
        v1, v2 = numpy.sum(((b1 - m1) ** 2) * p1) / q1, numpy.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
    # find otsu's threshold value with OpenCV function
    ret, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu



def selective_gaussian_filter(img: numpy.ndarray, d, strength) -> numpy.ndarray:
    """
    returns a copy of the given image with ana adaptive gaussian filter applied
    :param strength: strength is the strength of the filter
    :param d: diameter of the pixel neighborhood used during filtering
    :param img: input image to be filtered
    :return: filtered image

    --notes--
     d > 5 is very slow
    strength sets the value for both sigmacolor and sigmaspace
    TODO change default value for d and strength according to what seems good for experiment
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=strength, sigmaSpace=strength)




def sharpen(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = numpy.maximum(sharpened, numpy.zeros(sharpened.shape))
    sharpened = numpy.minimum(sharpened, 255 * numpy.ones(sharpened.shape))
    sharpened = sharpened.round().astype(numpy.uint8)
    if threshold > 0:
        low_contrast_mask = numpy.absolute(image - blurred) < threshold
        numpy.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened



"""
---NOTES---
I don't know what to do.

Well So what is image callibration???

Can I do a

First of all, I need a scoring function that takes  a

"""



def experiment_with_adaptive_filters():
    path2 = r"/Users/aidanlear/Desktop/qrcodes png/qr4.png"
    path2 = r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/curvedpics/qr5mod2.png"


    gridqr = cv2.imread(path2, 0)
    n = 5

    """
    boom = adaptive_gaussian(gridqr, 15)
    verification = qrcode2.verify_image(boom, n, show_unreadable_codes=True)
    percent = verification[0]
    print(f'Percent squares recognized={percent}')
    """

    for c in range(10):
        print(f"\n\nc value = {c}")
        for i in range(11, 37, 2):
            threshed_image = adaptive_mean(gridqr, i, c)
            verification = qrcode2.verify_image(threshed_image, n)
            percent = round(verification[0] * 100)

            print(f'block_size={i};  Percent squares recognized={percent}')




def experiment_with_gaussian_filters():
    path2 = r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/qrcodepics/qr5.png"
    #path2 = r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/curvedpics/qr5mod2.png"


    gridqr = cv2.imread(path2, 0)
    n = 5

    """
    boom = adaptive_gaussian(gridqr, 15)
    verification = qrcode2.verify_image(boom, n, show_unreadable_codes=True)
    percent = verification[0]
    print(f'Percent squares recognized={percent}')
    """

    for c in range(6):
        print(f"\nc value = {c}")
        for i in range(11, 37, 2):
            threshed_image = adaptive_gaussian(gridqr, i, c)
            verification = qrcode2.verify_image(threshed_image, n)
            percent = round(verification[0] * 100)
            if percent == 100:
                print(f'block_size={i};  Percent squares recognized={percent}')


def experiment_with_otsu():
    """testing otsu's binarization"""
    #read image into memory and apply apply binarization
    gridqr = cv2.imread(r"/Users/aidanlear/Desktop/qrcodes png/qr4.png", 0)
    binarized_image = otsu(gridqr)
    show_ndarray(binarized_image)

    verification_results = qrcode2.verify_image(binarized_image, 4)
    print("Results for 4 x 4:", verification_results)



    #now do this with a 3 x 3 image







def main():
    """entry point; only for testing purposes in this file"""
    #experiment_with_otsu()
    experiment_with_adaptive_filters()
    #experiment_with_gaussian_filters()










if __name__ == '__main__':
    main()
