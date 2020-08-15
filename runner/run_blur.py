import cv2
import time
import argparse
import sys, os
sys.path.append(os.pardir)
from mycv.blur import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--size", help="the window size of FastMeanBlur or the Gaussiankernel size of GaussBlur, \
                    the default value is 5", default=5, type=int)
    parser.add_argument("-v", "--variance", help="the value of the variance of Gaussian distribution when using \
                    GaussBlur, the default value is 1", default=1, type=int)
    parser.add_argument("img", help="the image to process")
    parser.add_argument("method", help="the blur method: FastMeanBlur/GaussBlur",choices=["FastMeanBlur","GaussBlur"])
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    cv2.imwrite(os.path.abspath('..')+"\\result\\blur\\o_"+now+".jpeg",img)
    if args.method == "FastMeanBlur" :
        start = time.time()
        result = FastMeanBlur.FastMeanBlur(img,(args.size,args.size))
        cv2.imwrite(os.path.abspath('..')+"\\result\\blur\\f_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "GaussBlur" :
        start = time.time()
        result = GaussBlur.GaussBlur(img,(args.size,args.size),args.variance,args.variance)
        cv2.imwrite(os.path.abspath('..')+"\\result\\blur\\g_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")

if __name__ == '__main__':
    main()


