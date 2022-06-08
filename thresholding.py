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

img = cv2.medianBlur(gray2,5)
ret,th1 = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]
for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()