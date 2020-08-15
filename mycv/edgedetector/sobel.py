# This is the Sobel Edge Detection algorithm
from scipy.signal import convolve2d
import numpy as np

'''
The Sobel Operator is a combination of the smoothing 
factor and the edge-enhancing factor. "1, 4, 6, 4, 1"
 are the binomial coefficients, used for smoothing; 
"1, 2, 0, -2, -1" is the differential coefficients, used
for edge-enhancing. The fast way is to split the kernel into
two separate arrays.
'''

''' 
differential factor on x, while smoothing factor on y. 
'''
def SobelxFast(img):
    R1 = np.array([[1], [4], [6], [4], [1]], np.float64)
    IconR1 = convolve2d(img, R1, "same")
    R2 = np.array([[1, 2, 0, -2, -1]], np.float64)
    IconR1 = convolve2d(IconR1, R2, "same")
    return IconR1

''' 
differential factor on y, while smoothing factor on x. 
'''
def SobelyFast(img):
    R1 = np.array([[1], [2], [0], [-2], [-1]], np.float64)
    IconR1 = convolve2d(img, R1, "same")
    R2 = np.array([[1, 4, 6, 4, 1]], np.float64)
    IconR1 = convolve2d(IconR1, R2, "same")
    return IconR1

'''
After the process, some pixels will have values over 255, or less than 0. 
Normalization is need to be done to bring the grayscale value back to 
[0, 255]. This is a linear normalization method.
'''
def normalLinear(I):
    M = np.max(I)
    ratio = 255 / M 
    I = ratio * I
    return I

'''
This is a nonlinear Gamma normalization method. This will incline to the 
pixels with smaller grayscale values, which will work well when the 
image is generally dim.
'''
def normalGamma(I, gamma):
    M = np.max(I)
    ratio = 255 / M
    I = ratio * I
    I = I / 255
    I = np.power(I, gamma)
    return I * 255

'''
This is the averaging of Sobel results on horizontal and vertical sides.
'''
def SobelFast(img): 
    img = np.sqrt(SobelxFast(img) ** 2 + SobelyFast(img) ** 2)
    img = normalGamma(img, 0.5)
    # img = normalLinear(img)
    return img

'''
By obtaining the complement of the result, we can get the pencil 
view of the original image.
'''
def pencil(img):
    return 255 - SobelyFast(img)