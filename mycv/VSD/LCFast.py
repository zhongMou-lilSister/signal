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
Linear normalization
'''
def normalLinear(I):
    M = np.max(I)
    N = np.min(I)
    I = 255 * (I - N) / (M - N)

    return I

'''
This is the faster way of computing LC saliency, with 
a complexity of O(256MN). It uses the grayscale histogram
to accelerate the process. But the downside is having traversing
256 grayscales for every pixel in the image.
'''
def LCFast(img):
    array = grayHist(img)
    ctr0 = 0
    ctr1 = 0
    M = len(img)
    N = len(img[0])
    result = np.zeros((M, N))

    for i in range(M):
        for j in range(N):
            for k in range(256):
                result[i][j] += array[k] * abs(k - img[i][j])

            if ctr1 - ctr0 > 0.01 * M * N:
                ctr0 = ctr1
                # print(ctr0 / (M * N))

            ctr1 += 1
    return normalLinear(result)