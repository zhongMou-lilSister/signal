import cv2
import time
import argparse
import sys, os
sys.path.append(os.pardir)
from mycv.threshold import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="the image to process")
    parser.add_argument("method", help="the threshold method: otsu/histogram/adaptiveThresh",
                        choices=["otsu","histogram","adaptiveThresh"])
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    cv2.imwrite(os.path.abspath('..')+"\\result\\threshold\\o_"+now+".jpeg",img)

    if args.method == "otsu" :
        start = time.time()
        result = Threshold.otsu(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\threshold\\os_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "histogram" :
        start = time.time()
        result = Threshold.histogram(img)
        cv2.imwrite(os.path.abspath('..')+"\\result\\threshold\\h_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")
    elif args.method == "adaptiveThresh" :
        start = time.time()
        result = Threshold.adaptiveThresh(img, img.shape[:2][::-1])
        cv2.imwrite(os.path.abspath('..')+"\\result\\threshold\\a_"+now+".jpeg",result)
        end = time.time()
        print("Execution Time: " + str(end-start) + "s")


if __name__ == '__main__':
    main()
