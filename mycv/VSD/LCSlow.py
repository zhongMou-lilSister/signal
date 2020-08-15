import numpy as np 

'''
Linear normalization
'''
def normalLinear(I):
    M = np.max(I)
    N = np.min(I)
    I = 255 * (I - N) / (M - N)

    return I

'''
This is the slowest method of all LC methods. It uses the
LC defination directly. For an image of M*N size, the complexity is
O(M^2N^2). For a method this slow, only 50*50 pixels can be dealt in 
a instant.
'''
def LCSlow(img):
    M = len(img)
    N = len(img[0])
    ctr0 = 0
    ctr1 = 0
    result = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            result[i][j] = 0
            
            for p in range(M):
                for q in range(N):
                    if not (p == i and q == j):
                        result[i][j] += abs(int(img[p][q]) - int(img[i][j]))
            
            if ctr1 - ctr0 > 0.01 * M * N:
                ctr0 = ctr1
                # print(ctr0 / (M * N))

            ctr1 += 1
    
    return normalLinear(result)
