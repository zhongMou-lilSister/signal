import sys, os
sys.path.append(os.pardir)
import cv2
import numpy as np


'''
This is the classic FT algorithm mentioned in the thesis.
It uses LAB color space to average, because LAB is more similar to
the biolaogical feature of human eyes than RGB.
'''
def FT(src):
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    img = cv2.GaussianBlur(img,(5, 5), 1)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    # temp = np.zeros((len(img), len(img[0])))
    l = lab[:,:,0]
    lmean = np.mean(lab[:,:,0])
    a = lab[:,:,1]
    amean = np.mean(lab[:,:,1])
    b = lab[:,:,2]
    bmean = np.mean(lab[:,:,2])
    
    ld = l - lmean
    ad = a - amean
    bd = b - bmean
    pic = ld * ld + ad * ad + bd * bd
    max_ = max([np.max(i) for i in pic])
    # linear normalization
    return pic * 255 / max_
