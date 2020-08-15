import numpy as np
import scipy.signal

#Get the Gaussian kernel
def GaussKernel(sigma,H):
	gausscol = np.zeros((H,1),np.float32)
	for r in range(H):
		gausscol[r][0] = np.exp(-0.5*((r-0.5*(H-1))**2)/(sigma)**2)
	sumGC = np.sum(gausscol)
	gausscol = gausscol/sumGC
	return gausscol

def GaussBlur(image,ksize=(5,5),sigmaX=1,sigmaY=1,bordertype='fill',fillvalue=0):
	#Convolution with X axis
	gausskernel_X = GaussKernel(sigmaX,ksize[1])
	gausskernel_X = np.transpose(gausskernel_X)
	gauss_result = scipy.signal.convolve2d(image,gausskernel_X,'same',
		bordertype,fillvalue)
	#Convolution with Y axis
	gausskernel_Y = GaussKernel(sigmaY,ksize[0])
	gauss_result = scipy.signal.convolve2d(gauss_result,gausskernel_Y,
		'same',bordertype,fillvalue)
	return gauss_result
