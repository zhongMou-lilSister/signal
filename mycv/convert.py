'''
This is the 2D -> 1D spectrum averaging process. it takes in a 
grayscale image and outputs the digital angular frequency and the 
amplitude arrays of the image signal0, for later plotting purposes.
'''
import numpy as np

def Convert(img):
    # data preparation, FFT and log
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    s2 = np.log(np.abs(fshift) + 1)
    xc = int(len(img) / 2)
    yc = int(len(img[0]) / 2)
    # divide img into quarters 
    l = int(np.sqrt(xc**2 + yc**2))
    C = int(l / 100) * 100
    # get the memory usage ready
    arr = np.zeros(C)
    record = np.zeros(C)
    #extract info in O(MN/4)
    for i in range(xc):
        for j in range(yc):
            thisL = int(C*np.sqrt((i/xc)**2 + (j/yc)**2)/1.414)
            thisAmp = s2[xc+i][yc+j]
            arr[thisL] += thisAmp
            record[thisL] += 1
    for i in range(C):
        # rule out the denominator = 0 situations.
        if record[i] == 0:
            record[i] == 1
    # average over
    arr /= record
    return (arr, C)
