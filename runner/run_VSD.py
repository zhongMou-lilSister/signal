import cv2
import time
import argparse
import sys, os
sys.path.append(os.pardir)
from mycv.VSD import *



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="the image to process")
    parser.add_argument("method", help="the VSD method: FT/HC/LCFast/LCFaster/LCSlow",
                        choices=["FT","HC","LCFast","LCFaster","LCSlow"])
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_COLOR)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\o_"+now+".jpeg",img)
    if args.method == ("FT" or "HC") :
        if args.method == "FT" :
            start = time.time()
            result = FT.FT(img)
            cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\FT_"+now+".jpeg",result)
            end = time.time()
            print("Execution Time: " + str(end-start) + "s")
        elif args.method == "HC" :
            start = time.time()
            result = HC.HC(img)
            cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\HC_"+now+".jpeg",result)
            end = time.time()
            print("Execution Time: " + str(end-start) + "s")

    else:
        img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
        if args.method == "LCFast":
            start = time.time()
            result = LCFast.LCFast(img)
            cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\LCF_"+now+".jpeg",result)
            end = time.time()
            print("Execution Time: " + str(end-start) + "s")
        elif args.method == "LCFaster":
            start = time.time()
            result = LCFaster.LCFaster(img)
            cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\LCFer_"+now+".jpeg",result)
            end = time.time()
            print("Execution Time: " + str(end-start) + "s")
        elif args.method == "LCSlow":
            start = time.time()
            result = LCSlow.LCSlow(img)
            cv2.imwrite(os.path.abspath('..')+"\\result\\VSD\\LCS_"+now+".jpeg",result)
            end = time.time()
            print("Execution Time: " + str(end-start) + "s")

if __name__ == '__main__':
    main()
