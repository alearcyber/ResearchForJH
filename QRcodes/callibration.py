"""
TODO fix header comment

References
dual_annealing -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html
optimizeResult object -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult

"""

import cv2
import numpy
import time
import thresh
import qrcode2
import itertools
import scipy.optimize as sp



#dimensions of the qr code grid
N = None

#the sample image being calibrated against
IMAGE = None

# kernel is KERNEL x KERNEL
KERNEL = 3







def sharpen_hill_climb():
    #result = sp.dual_annealing(process_image, bounds=[[0, 1]], maxiter=100)

    #INTIAL GUESS is x0
    i = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)
    x0 = [0.5, 0.5]
    N = 5
    tolerance = 100/(N**2)

    #result = sp.minimize(process_image2, method='Nelder-Mead', x0=x0, bounds=[[0.1, 2], [0.1, 2]], tol=tolerance)     # sigma, amount
    #result = sp.dual_annealing(process_image2, bounds=[[0, 2], [0, 2]], maxiter=10, )
    #result = sp.differential_evolution(process_image2, x0=x0, bounds=[[0.1, 2], [0.1, 2]], tol=20, maxiter=50, disp=True)
    #result = sp.fmin(process_image2, )


    #input
    kernel_values = [1, 3]
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    amount_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    N = 5


    best_percent = 0

    best_values = {'kernel': None, 'sigma': None, 'amount': None}

    for kernel in kernel_values:
        for sigma in sigma_values:
            for amount in amount_values:
                sharpened = thresh.sharpen(i, kernel_size=(kernel, kernel), sigma=sigma, amount=amount)
                percent = qrcode2.verify_image_percent(sharpened, N)
                if percent == 100:
                    return best_values, 100
                elif percent > best_percent:
                    best_values['kernel'] = kernel
                    best_values['sigma'] = sigma
                    best_values['amount'] = amount


    return best_values, best_percent



def time_trial_brute_force():
    """
    finds the best variables for sharpneing with a brute force method.
    Prints the time it takes
    """

    test_image = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)
    N = 5  #nxn qr code grid


    #possible range of values
    kernel_values = [1, 3]
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    amount_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]


    #store the best percent of qr codes identified, along with its corresponding input variables
    best_percent = 0
    best_values = {'kernel': None, 'sigma': None, 'amount': None}
    found_best_flag = False  #flag to enabling breaking both inner and outter loop

    start_time = time.time()  # starting time
    print('Starting Brute Force...')

    #Iterate over the find the best values
    for kernel, sigma, amount in itertools.product(kernel_values, sigma_values, amount_values):
        sharpened = thresh.sharpen(test_image, kernel_size=(kernel, kernel), sigma=sigma, amount=amount)
        percent = qrcode2.verify_image_percent(sharpened, N)
        if percent == 100:
            best_values['kernel'] = kernel
            best_values['sigma'] = sigma
            best_values['amount'] = amount
            best_percent = percent
            break
        elif percent > best_percent:
            best_values['kernel'] = kernel
            best_values['sigma'] = sigma
            best_values['amount'] = amount
            best_percent = percent





    #print the time elapsed and values found
    print('Brute Force time elapsed:', time.time() - start_time)
    best_kernel = best_values['kernel']
    best_sigma = best_values['sigma']
    best_amount = best_values['amount']
    print(f'kernel:{best_kernel}  sigma:{best_sigma}  amount:{best_amount}  percent:{best_percent}')


    #verify the results
    sharpened_verification_image = thresh.sharpen(test_image, kernel_size=(best_kernel, best_kernel), sigma=best_sigma, amount=best_amount)
    test_result_percent = qrcode2.verify_image_percent(sharpened_verification_image, N)
    if test_result_percent == best_percent:
        print('Results Check out')
    else:
        print("There is an error the values found")





def time_trial_differential_evolution():
    """
    time trial for finding varibles with differential evolution
    """
    #initial guess is x0
    initial_guess = [0.5, 0.5]

    print('Starting Differential Evolution...')
    start_time = time.time()
    result = sp.differential_evolution(process_image2, x0=initial_guess, bounds=[[0.1, 2], [0.1, 2]], tol=20, maxiter=20)
    print('Differential Evolution time elapsed:', time.time() - start_time)
    print('results\n', result)


def time_trial_dual_annealing():
    """
    time trial for finding variables with dual annealing
    """
    #constants
    test_image = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)
    N = 5  # nxn qr code grid


    #initial guess is x0
    initial_guess = [0.5, 0.5]
    start_time = time.time()
    print('Starting Dual Annealing...')
    dual_annealing_results = sp.dual_annealing(process_image2, x0=initial_guess, bounds=[[0, 2], [0, 2]], maxiter=100, callback=dual_anneal_callback)
    print('Dual Annealing time elapsed:', time.time() - start_time)

    #parse out the results
    sigma_found = dual_annealing_results.x[0]
    amount_found = dual_annealing_results.x[1]
    percent_missing = dual_annealing_results.fun


    #print out results
    print(f'Inputs found: sigma={sigma_found}, amount={amount_found}, percent missing={percent_missing}')


    #test to ensure those results actually produce 100% of qr codes found
    sharpened_image_for_verification = thresh.sharpen(test_image, kernel_size=(3, 3), sigma=sigma_found, amount=amount_found)
    test_result_percent = qrcode2.verify_image_percent(sharpened_image_for_verification, N)
    if test_result_percent == 100:
        print('Results Check out')
    else:
        print("There is an error the values found")


def dual_anneal_callback(x, f, content):
    """
    This comment is Directly Taken from the documentation....

    called for all minima found. x and f are the coordinates and function value.
    Context has value in [0, 1, 2], with the following meaning:
        0: minimum detected in the annealing process.
        1: detection occurred in the local search process.
        2: detection done in the dual annealing process.
    Stops if this function returns true
    """
    if f == 0:
        return True

def time_trial_minimize():
    """
    TODO comment this better
    """
    initial_guess = [1.0, 1.0]

    start_time = time.time()
    print('starting')
    options = {'disp': True}
    minimize_results = sp.minimize(process_image2, x0=initial_guess, method='powell', bounds=[[0.1, 2], [0.1, 2]],
                                   callback=minimize_callback, options=options)


    print('\n\ndone:', time.time() - start_time)
    print(minimize_results)


def minimize_callback(params, state):
    print('-----------------------------------')
    print("params:", params, "     State:", state)




def process_image2(params):
    """
    a different version of the function to be minimized that has 3 parameters.
    Uses a 3,3 kernel
    """
    i = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)
    sigma, amount = params # unpack parameters
    sharpened = thresh.sharpen(i, kernel_size=(3, 3), sigma=sigma, amount=amount)
    return 100 - qrcode2.verify_image_percent(sharpened, 5)











def calibrate(img_in: numpy.ndarray, n_in: int, initial_guess=(0.5, 0.5)):
    """
    :param img_in: numpy.ndarray -> input image to be calibrated against.
    :param n_in: int -> n_in x n_in grid of qr codes contained within the image
    :param initial_guess: -> initial guess values for sigma and amount. Default is (0.5, 0.5)
    :return: 2-tuple of floats with the best resulting values, specifically sigma and amount
    """
    #Note: the setting of these global variables is required for ensuring the function being minimized
    #correctly assesses the image.
    #set the global constant for the dimensions of the qr code grid within the image
    global N
    N = n_in

    #set global constant for the input image
    global IMAGE
    IMAGE = img_in

    #perform the dual annealing
    dual_annealing_results = sp.dual_annealing(process_image, x0=initial_guess, bounds=[[0, 2], [0, 2]], maxiter=50,
                                               callback=calibrate_callback)

    #parse out results
    best_sigma = dual_annealing_results.x[0]
    best_amount = dual_annealing_results.x[1]
    percent_missing = dual_annealing_results.fun

    #verify the results
    sharpened_image_for_verification = thresh.sharpen(IMAGE, kernel_size=(KERNEL, KERNEL), sigma=best_sigma, amount=best_amount)
    test_percent = qrcode2.verify_image_percent(sharpened_image_for_verification, N)
    assert test_percent == 100, f'ERROR: was unable to find parameters that made all the qr codes readable.\n' \
                                f'Current parameters: sigma={best_sigma}, amount={best_amount}, percent missing={percent_missing}'

    #return the sigma and amount values found as a 2-tuple
    return best_sigma, best_amount


def calibrate_callback(x, f, content):
    """
    Callback function called in the dual annealing process for calibrate().
    Rest of this comment is directly taken from the documentation....

    called for all minima found. x and f are the coordinates and function value.
    Context has value in [0, 1, 2], with the following meaning:
        0: minimum detected in the annealing process.
        1: detection occurred in the local search process.
        2: detection done in the dual annealing process.
    Stops if this function returns true
    """
    return f == 0


def process_image(params):
    """
    this is the function to be minimized.
    It returns the percent of qr codes missed from the photo.
    """
    sigma, amount = params
    sharpened = thresh.sharpen(IMAGE, kernel_size=(KERNEL, KERNEL), sigma=sigma, amount=amount)
    return 100 - qrcode2.verify_image_percent(sharpened, N)


def test_calibrate():
    """
    test function for ensuring calibration operates correctly
    """
    #grab test image, it is 5 x 5
    test_image = cv2.imread(r"/Users/aidanlear/PycharmProjects/VCResearch/QRcodes/sharpened-qrcodes/qr5.png", 0)

    print('testing calibration...')
    start_time = time.time()
    sigma, amount = calibrate(test_image, 5)
    print('DONE! Time elapsed:', time.time() - start_time)
    print(f'Sigma={sigma}, Amount={amount}')


if __name__ == '__main__':
    """
    sigma_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    amount_values = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    for sigma in sigma_values:
        for amount in amount_values:
            percent_missed = process_image2([sigma, amount])
            print(f'Sigma:{sigma}, amount:{amount}, Percent Missed:{percent_missed}')
    """

    #basic implementation of communicating with named
    #try callback function
    #take grab still with opencv or whqtev3er else
    #come back monday

    test_calibrate()

