from __future__ import print_function
from nis import match
import cv2 
import numpy as np
from matplotlib import pyplot as plt
import argparse
import os

# Parse arguments
parser = argparse.ArgumentParser(description='Code for detecting peeled areas in vegetables')
parser.add_argument('--folder', help='Path to input images.', default='potato')
parser.add_argument('--image', help='Name of image to process. If empty all images are processed.', default='')
args = parser.parse_args()

# Iterate through folder or select specific image
path = os.path.join("./img_data/", args.folder)
images =  next(os.walk(path), (None, None, []))[2] if args.image == '' else args.image 

for image in images:

    # Load query and train images
    image = os.path.join(path, image)
    img = cv2.imread(image) 
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img is None:
        print('Could not open or find the images!')
        exit(0)
    print(f"Image Shape: {img.shape}")

    ###########  Process image ###########

    # Crop sides
    h, w = img.shape
    minW, maxW = int(w/10), int(9*w/10)
    print(f"Min: {minW} Max: {maxW}")
    crop1 = img[:,minW:maxW]
    print(crop1.shape)

    # Normalize image to min and max intensity of 0 and 255
    print(f"Crop min: {np.min(crop1)} max: {np.max(crop1)}")
    norm = np.zeros_like(crop1)
    norm = cv2.normalize(crop1, norm, 0, 255, cv2.NORM_MINMAX)
    print(f"Norm min: {np.min(norm)} max: {np.max(norm)}")

    # Segment
    blur = cv2.medianBlur(norm,5)
    ret,th1 = cv2.threshold(blur,140,255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(blur,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
    # Otsu's thresholding
    ret2,th4 = cv2.threshold(blur, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur2 = cv2.GaussianBlur(blur, (5,5),0)
    ret3,th5 = cv2.threshold(blur2,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    titles = ['Original Image', 'Crop', 'Norm', 'Blur', 'Global Thresholding (v = 127)',
                'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', 'Otsu', 'Otsu w/ Blur']
    images = [img, crop1, norm, blur, th1, th2, th3, th4, th5]

    # Display image
    for i in range(9):
        plt.subplot(3,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()