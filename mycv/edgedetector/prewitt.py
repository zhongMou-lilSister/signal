# This is the Prewitt Edge Detection algorithm
from scipy.signal import convolve2d
import numpy as np

'''
There are two kinds of Prewitt algorithm, a fast one and a slow one.
The difference between the two kinds depends on the size of the Prewitt 
Operator, a.k.a, the kernel(with the size of k). The fast one is O(MNk), 
while the slow one is O(MNk^2). The key is the disassemble of the kernel.
'''

'''
The slow version that emphasize on the difference of the horizontal direction. 
'''
def prewittxSlow(img):
    R1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], np.float64)
    IconR1 = convolve2d(img, R1, 'same')
    return IconR1

'''
The slow version that emphasize on the difference of the vertical direction. 
'''
def prewittySlow(img):
    R2 = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], np.float64)
    IconR2 = convolve2d(img, R2, 'same')
    return IconR2

'''
The fast version that emphasize on the difference of the horizontal direction. 
'''
def prewittxFast(img):
    R1 = np.array([[1], [1], [1]], np.float64)
    IconR1 = convolve2d(img, R1, 'same')
    R2 = np.array([[1, 0, -1]], np.float64)
    IconR1 = convolve2d(IconR1, R2, 'same')
    return IconR1

'''
The fast version that emphasize on the difference of the vertical direction. 
'''
def prewittyFast(img):
    R1 = np.array([[1, 1, 1]], np.float64)
    IconR1 = convolve2d(img, R1, 'same')
    R2 = np.array([[1], [0], [-1]], np.float64)
    IconR1 = convolve2d(IconR1, R2, 'same')
    return IconR1

'''
The slow version, averaging the two directions. 
'''
def prewittSlow(img):
    return np.sqrt(prewittxSlow(img) ** 2 + prewittySlow(img) ** 2)

'''
The fast version, averaging the two directions. 
'''
def prewittFast(img):
    return np.sqrt(prewittxFast(img) ** 2 + prewittyFast(img) ** 2)