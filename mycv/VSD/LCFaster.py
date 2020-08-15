import numpy as np 

'''
The calculation of the grayscale histogram of 
a certain image.
'''
def grayHist(img):
    array = np.zeros(256)
    M = len(img)
    N = len(img[0])
    for i in range(M):
        for j in range(N):
            array[int(img[i][j])] += 1

    array /= (M * N)
    return array

'''
The calculation of the ratio of overall 
grayscale. O(256^2)
'''
def calcDist(array):
    table = np.zeros(256)
    for i in range(256):
        for j in range(256):
            table[i] += abs(i - j) * array[j]
    
    return table

'''
Linear normalization.
'''
def normalLinear(I):
    M = np.max(I)
    N = np.min(I)
    I = 255 * (I - N) / (M - N)

    return I

'''
This is the fastest way of computing LC saliency, 
with a complexity of O(256^2+MN). The histogram actually 
has the ratio information within, so an addtional 
iteration in traversing every pixel is not necessary. 
'''
def LCFaster(img):
    array = grayHist(img)
    table = calcDist(array)
    M = len(img)
    N = len(img[0])
    result = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            result[i][j] = table[int(img[i][j])]

    return normalLinear(result)