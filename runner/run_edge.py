import cv2
import time
import argparse
import sys, os
sys.path.append(os.pardir)
from mycv.edgedetector import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="the image to process")
    parser.add_argument("method", help="the edge detector method: fastprewitt/slowprewitt/roberts/sobel",
                        choices=["fastprewitt","slowprewitt","roberts","sobel"])
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\o_"+now+".jpeg",img)
    if args.method == "fastprewitt" :
        start = time.time()
        result = prewitt.prewittFast(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\fp_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "slowprewitt" :
        start = time.time()
        result = prewitt.prewittSlow(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\sp_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "roberts" :
        start = time.time()
        result = roberts.roberts(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\r_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "sobel" :
        start = time.time()
        result = sobel.SobelFast(img)
        result1 = sobel.pencil(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\s_"+now+".jpeg",result)
        cv2.imwrite(os.path.abspath('..')+"\\result\\edge_detector\\s_pv_"+now+".jpeg",result1)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")

if __name__ == '__main__':
    main()
