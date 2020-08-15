import cv2
import numpy as np
# otsu Threshold segmentation
def otsu(img):
    width,height=img.shape[:2][::-1]
    # Get gray histogram
    grayhist=cv2.calcHist([img],[0],None,[256],[0,255])
    # Histogram normalization
    uniformgrayhist=grayhist/float(width*height)
    # Calculate the zero order cumulative moment and the first order cumulative moment
    ZeroCM=np.zeros([256],np.float32)
    OneCM=np.zeros([256],np.float32)
    for k in range(256):
        if k==0:
            ZeroCM[k]=uniformgrayhist[0]
            OneCM[k]=k*uniformgrayhist[0]
        else:
             ZeroCM[k]=ZeroCM[k-1]+uniformgrayhist[k]
             OneCM[k]=OneCM[k-1]+k*uniformgrayhist[k]
    # Calculate variance
    var=np.zeros([256],np.float32)
    for k in range(255):
        if ZeroCM[k]==0 or ZeroCM[k]==1:
            var[k]=0
        else:
            var[k]=pow(OneCM[255]*ZeroCM[k]-OneCM[k],2)/(ZeroCM[k]*(1.0-ZeroCM[k]))
    # Search threshold
    location=np.where(var[0:255]==np.max(var[0:255]))
    thresh=location[0][0]
    threshimage_out1=np.copy(img)
    threshimage_out1[threshimage_out1>thresh]=255
    threshimage_out1[threshimage_out1<=thresh]=0
    return threshimage_out1
# Histogram threshold segmentation
def histogram(image):
    # Get gray histogram
    histogram=cv2.calcHist([image],[0],None,[256],[0,255])
    # Find the gray value corresponding to the first peak
    max1=np.where(histogram==np.max(histogram))
    peak1=max1[0][0]
    # Find the gray value corresponding to the second peak
    histogram2=np.zeros([256],np.float32)
    for k in range(256):
        histogram2[k]=pow(k-peak1,2)*histogram[k]
    max2=np.where(histogram2==np.max(histogram2))
    peak2=max2[0][0]
    if peak1>peak2:
        temp=histogram[int(peak2):int(peak1)]
        min=np.where(temp==np.min(temp))
        thresh=peak2+min[0][0]+1
    else:
        temp=histogram[int(peak1):int(peak2)]
        min=np.where(temp==np.min(temp))
        thresh=peak1+min[0][0]+1
    # Threshold processing
    threshimage_out2=image.copy()
    threshimage_out2[threshimage_out2>thresh]=255
    threshimage_out2[threshimage_out2<=thresh]=0
    return threshimage_out2

#Adaptive threshold segmentation
def adaptiveThresh(ip,wh):
    mean=cv2.boxFilter(ip,-1,wh)
    threshimage_out3=ip-0.85*mean
    threshimage_out3[threshimage_out3>=0]=255
    threshimage_out3[threshimage_out3<0]=0
    threshimage_out3=threshimage_out3.astype(np.uint8)
    return threshimage_out3

