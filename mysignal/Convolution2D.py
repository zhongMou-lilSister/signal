import numpy as np

def convolution(image,ckernel,mode):
   H,W=ckernel.shape
   h,w=image.shape #convolution kernel's size and image's size
   ckernel=np.flipud(ckernel)
   ckernel=np.fliplr(ckernel) #rotate the picture 180 degrees counterclockwise
   modelmatrix=np.pad(image,((H-1,H-1),(W-1,W-1)),'constant') #use constant 0 to fill around the matrix
   full_matrix = np.empty((h+H-1, w+W-1))
   for i in range(0,H+h-1):
      for j in range(0,W+w-1):
         s_matrix=modelmatrix[i:i+H,j:j+W]
         m_matrix = np.array(ckernel)* np.array(s_matrix)
         full_matrix[i,j] = np.sum(m_matrix) #full_convolution matrix
   if (H%2)==1:
      mH=(H-1)/2
   else:
      mH=H/2
   if (W%2)==1:
      mW=(W-1)/2
   else:
      mW=W/2
   same_matrix=full_matrix[int(H-mH-1):int(H-mH-1+h),int(W-mW-1):int(W-mW-1+w)] #choose appropriate point to cut out a section of matrix, which is the same_convolution matrix
   if mode=='full':
      return (full_matrix)
   elif mode =='same':
      return (same_matrix)
