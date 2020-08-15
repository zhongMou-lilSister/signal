import cv2
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.pardir)
from mycv import convert



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("img", help="the image to process")
    args = parser.parse_args()

    img = cv2.imread(args.img, cv2.IMREAD_GRAYSCALE)
    now = time.strftime("%Y_%m_%d_%H_%M_%S")
    cv2.imwrite(os.path.abspath('..')+"\\result\\convert\\o_"+now+".jpeg",img)

    result = convert.Convert(img)
    amp = result[0]
    ang = np.linspace(0.0, 3.1415926, num=result[1])
    plt.plot(ang, amp)
    plt.xlabel("Angular Frequency")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.savefig(os.path.abspath('..')+"\\result\\convert\\spectrum_"+now+".jpeg")

if __name__ == '__main__':
    main()
