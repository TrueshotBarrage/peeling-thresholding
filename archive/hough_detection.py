from __future__ import print_function
from nis import match
import cv2 as cv2
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
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.imread(args.train) 
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
if img1 is None or img2 is None:
    print('Could not open or find the images!')
    exit(0)

# Find the edges in the image using canny detector
edges = cv2.Canny(gray2, 50, 200)
# Detect points that form a line
lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=10, maxLineGap=20)
# Draw lines on the image
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img2, (x1, y1), (x2, y2), (255, 0, 0), 3)
# Show result
cv2.imshow("Result Image", img2)
