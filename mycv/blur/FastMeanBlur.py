import numpy as np 

def FastMeanBlur(image,winsize=(5,5)):
	H_half = int((winsize[0]-1)/2)
	W_half = int((winsize[1]-1)/2)
	image_border = borderzero(image,H_half,W_half)
	image_inte = integral(image_border)
	#Adding zeros to the left colum and the top row
	row,col = image_inte.shape
	image_do = np.zeros((row+1,col+1),np.float32)
	image_do[1:1+row,1:1+col] = image_inte
	#Calculate the result
	row,col = image.shape
	image_result = np.zeros(image.shape,np.float32)
	for r in range(row):
		for c in range(col):
			image_result[r][c] = (image_do[r+1+H_half*2][c+1+W_half*2]+
				image_do[r][c]-image_do[r+1+H_half*2][c]-
				image_do[r][c+1+W_half*2])/(winsize[0]*winsize[1])

	return image_result


#Border processing(zeros)
def borderzero(image,H:int,W:int):
	row,col = image.shape
	border_image = np.zeros((row+2*H,col+2*W),np.float32)
	border_image[H:H+row,W:W+col] = image
	return border_image


#Image integral
def integral(image):
	row,col = image.shape
	inte_image = np.zeros((row,col),np.float32)
	#Colum integral
	for  c in range(col):
		for r in range(row):
			if r == 0:
				inte_image[r][c] = image[r][c]
			else:
				inte_image[r][c] = inte_image[r-1][c] + image[r][c]
	#Row integral
	for r in range(row):
		for c in range(col):
			if c != 0:
				inte_image[r][c] = inte_image[r][c-1] + inte_image[r][c]
	return inte_image
