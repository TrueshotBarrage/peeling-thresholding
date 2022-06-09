from __future__ import print_function
from nis import match
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Code for adaptive cutting board detection')
parser.add_argument('--query', help='Path to input image 1.', default='imgs/2.jpg')
parser.add_argument('--train', help='Path to input image 2.', default='imgs/2.jpg')
args = parser.parse_args()

# Load query and train images
img1 = cv2.imread(args.query) 
gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
img2 = cv2.imread(args.train) 
gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
# gray2 = cv2.convertScaleAbs(gray2)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

# global thresholding
ret1,th1 = cv2.threshold(gray2, 127,255, cv2.THRESH_BINARY)
# Otsu's thresholding
ret2,th2 = cv2.threshold(gray2, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(gray2, (5,5),0)
ret3,th3 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# plot all the images and their histograms
images = [gray2,  0, th1,
          gray2,  0, th2,
          blur, 0, th3]
titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
          'Original Noisy Image','Histogram',"Otsu's Thresholding",
          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])

plt.show()