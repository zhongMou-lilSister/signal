import sys, os
sys.path.append(os.pardir)
import numpy as np
import math
from mycv.blur import *

def HC(image):
	row,col,rgb = image.shape
	image = RGB12(image)
	#Calculate the color histograms
	color = np.zeros((12,12,12))
	for r in range(row):
		for c in range(col):
			color[image[r][c][0]-1][image[r][c][1]-1][image[r][c][2]-1] += 1
	#Calculate the saliency of each color
	dist = np.zeros((12,12,12))
	dif_b = np.zeros(12)
	dif_g = np.zeros(12)
	dif_r = np.zeros(12)
	for b in range(12):
		for cb in range(12):
			dif_b[b] += np.sum(np.sum(color, axis = 2), axis = 1)[cb] * abs(b-cb)
	for g in range(12):
		for cg in range(12):
			dif_g[g] += np.sum(np.sum(color, axis = 2), axis = 0)[cg] * abs(g-cg)
	for r in range(12):
		for cr in range(12):
			dif_r[r] += np.sum(np.sum(color, axis = 0), axis = 0)[cr] * abs(r-cr)
	for b in range(12):
		for r in range(12):
			for g in range(12):
				dist[b][g][r] = dif_b[b] + dif_g[g] + dif_r[r]

	'''
	for b in range(12):
		for g in range(12):
			for r in range(12):
				value = 0
				for cb in range(12):
					for cg in range(12):
						for cr in range(12):
							value += color[cb][cg][cr]*math.sqrt((b-cb)**2 + (g-cg)**2 + (r-cr)**2)
				dist[b][g][r] = value
	'''
	#Renew the image with the saliency and normalize
	image_result = np.zeros((row,col))
	temp = np.zeros(3)
	for r in range(row):
		for c in range(col):
			temp = image[r][c]
			image_result[r][c] = dist[temp[0]-1][temp[1]-1][temp[2]-1]
	image_result = 255*(image_result - np.min(image_result))/(np.max(image_result) - np.min(image_result))
	image_result = GaussBlur.GaussBlur(image_result,(2,2),0.67,0.67)
	return image_result



#256*256*256(RGB) to 12*12*12(RGB)
def RGB12(image):
	row,col,rgb = image.shape
	image_result = np.zeros(image.shape,np.int)
	for r in range(row):
		for c in range(col):
			image_result[r][c][0] = (math.ceil(image[r][c][0]/23))
			image_result[r][c][1] = (math.ceil(image[r][c][1]/23))
			image_result[r][c][2] = (math.ceil(image[r][c][2]/23))

	return image_result
