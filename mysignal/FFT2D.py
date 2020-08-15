import numpy as np

def DFT(sig):
    N = sig.size
    V = np.array([[np.exp(-1j*2*np.pi*v*y/N) for v in range(N)] for y in range(N)])
    return sig.dot(V)
def FFT(x):
    N = x.shape[1] #just consider the second dimension, and then loop through the first dimension
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 8:  # this cutoff should be optimized
        return np.array([DFT(x[i,:]) for i in range(x.shape[0])])
    else:
        X_even = FFT(x[:,::2])
        X_odd = FFT(x[:,1::2])
        factor = np.array([np.exp(-2j * np.pi * np.arange(N) / N) for i in range(x.shape[0])])
        return np.hstack([X_even + np.multiply(factor[:,:int(N/2)],X_odd),
                               X_even + np.multiply(factor[:,int(N/2):],X_odd)])
def FFT2D(img):
    return FFT(FFT(img).T).T
def FFT_SHIFT(img):
    M,N = img.shape
    M = int(M/2)
    N = int(N/2)
    return np.vstack((np.hstack((img[M:,N:],img[M:,:N])),np.hstack((img[:M,N:],img[:M,:N]))))
