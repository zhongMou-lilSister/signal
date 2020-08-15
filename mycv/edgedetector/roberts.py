# This is the Roberts Edge Detection algorithm

from scipy.signal import convolve2d
import numpy as np

'''
Roberts 45 degree operator
[[1, 0],
 [0, -1]]
After convolution, it will be upper-left pixel
minus lower-right pixel.
'''
def roberts45(img):
    R1 = np.array([[1, 0], [0, -1]], np.float64)
    IconR1 = convolve2d(img, R1)
    return IconR1

'''
Roberts 135 degree operator
[[0, 1],
 [-1, 0]]
After convolution, it will be upper-right pixel
minus lower-left pixel.
'''
def roberts135(img):
    R2 = np.array([[0, 1], [-1, 0]], np.float64)
    IconR2 = convolve2d(img, R2)
    return IconR2

'''
This is a way of averaging two styles of 
edge detecting methods.
'''
def roberts(img):
    return np.sqrt(roberts45(img) ** 2 + roberts135(img) ** 2)

