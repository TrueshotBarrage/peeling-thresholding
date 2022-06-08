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
images =  next(os.walk(path), (None, None, []))[2] if args.image == '' else [args.image]

# Loop through images and process
for imFile in sorted(images):

    # Load image and convert to gray
    print(f"Processing {imFile }")
    imPath = os.path.join(path, imFile)
    img = cv2.imread(imPath) 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        print('Could not open or find the images!')
        exit(0)
    print(f"Image Shape: {img.shape}")

    ###########  Process image ###########

    # Crop sides
    h, w, c = img.shape
    minW, maxW = int(w/10), int(9*w/10)
    print(f"Min: {minW} Max: {maxW}")
    # img = img[:, minW:maxW]
    print(img.shape)

    # Normalize image to min and max intensity of 0 and 255
    print(f"Crop min: {np.min(img)} max: {np.max(img)}")
    norm = np.zeros_like(img)
    norm = cv2.normalize(img, norm, 0, 255, cv2.NORM_MINMAX)
    print(f"Norm min: {np.min(norm)} max: {np.max(norm)}")

    # Get separate channels
    zeros = np.zeros_like(norm.shape[:2])
    (R, G, B) = cv2.split(norm)
    zeros = np.zeros(norm.shape[:2], dtype="uint8")
    red   = cv2.merge([R, zeros, zeros])
    green = cv2.merge([zeros, G, zeros])
    blue  = cv2.merge([zeros, zeros, B])
    gray = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)

    def segment(img):
        # global thresholding
        ret1,th1 = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th1, th2, th3

    i1, i2, i3 = segment(gray)
    r1, r2, r3 = segment(R)
    g1, g2, g3 = segment(G)
    b1, b2, b3 = segment(B)

    # Display
    titles = ['Original', 'Red', 'Green', 'Blue', 
              'Original - Gray', 'Red - Gray', 'Green - Gray', 'Blue - Gray',
              'Original - Global', 'Red - Global', 'Green - Global', 'Blue - Global', 
              'Original - Otsu', 'Red - Otsu', 'Green - Otsu', 'Blue - Otsu', 
              'Original - Otsu + Gauss', 'Red - Otsu + Gauss', 'Green - Otsu + Gauss', 'Blue - Otsu + Gauss']
    images = [img, red, green, blue, gray, R, G, B, i1, r1, g1, b1, i2, r2, g2, b2, i3, r3, g3, b3]
    # images = [img, gray, R, G, B]

    # Display image
    for i in range(20):
        plt.subplot(5,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    ###### Segment #######
    # Use blue channel as it seems to provide most accurate segmentations
    # Of the three methods, find the one where the masked area is smallest as to not include bleed

    def morph(img):
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((100,100),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img


    clean1 = morph(b1)
    clean2 = morph(b2)
    clean3 = morph(b3)

    img1, img2, img3 = np.copy(img), np.copy(img), np.copy(img)

    # Get mask
    ret, thresh = cv2.threshold(clean1, 127, 255, 0)
    contours, hierarchy = cv2.findContours(np.invert(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    cv2.drawContours(img1,    contours, 0, (0, 255, 0), 3)
    mask1 = np.zeros_like(img1[:,:,0])
    mask1 =  cv2.drawContours(mask1, contours, 0, (255), -1)
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(contours[0])
    # to save the images
    img1 = img1[y:y+h,x:x+w]
    mask1 = mask1[y:y+h,x:x+w]


    ret, thresh = cv2.threshold(clean2, 127, 255, 0)
    contours, hierarchy = cv2.findContours(np.invert(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    cv2.drawContours(img2,    contours, 0, (0, 255, 0), 5)
    mask2 = np.zeros_like(img2[:,:,0])
    mask2 =  cv2.drawContours(mask2, contours, 0, (255),-1)
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(contours[0])
    # to save the images
    img2 = img2[y:y+h,x:x+w]
    mask2 = mask2[y:y+h,x:x+w]

    ret, thresh = cv2.threshold(clean3, 127, 255, 0)
    contours, hierarchy = cv2.findContours(np.invert(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    cv2.drawContours(img3,    contours, 0, (0, 255, 0), 5)
    mask3 = np.zeros_like(img3[:,:,0])
    mask3 =  cv2.drawContours(mask3, contours, 0, (255),-1)
    # get the bounding rect
    x, y, w, h = cv2.boundingRect(contours[0])
    # to save the images
    img3 = img3[y:y+h,x:x+w]
    mask3 = mask3[y:y+h,x:x+w]

    thresh1, _, _ = segment(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    _, thresh2, _ = segment(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    _, _, thresh3 = segment(cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY))

    kernel = np.ones((10,10),np.uint8)
    mask1 = cv2.erode(mask1,kernel,iterations = 1)
    mask2 = cv2.erode(mask2,kernel,iterations = 1)
    mask3 = cv2.erode(mask3,kernel,iterations = 1)
    thresh1 = cv2.bitwise_and(mask1, thresh1)
    thresh2 = cv2.bitwise_and(mask2, thresh2)
    thresh3 = cv2.bitwise_and(mask3, thresh3)

    contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    # contours = contours[2:] if len(contours) > 2 else [] 
    print(len(contours))
    cv2.drawContours(img1,    contours, -1, (255, 0, 0), -1)

    contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    # contours = contours[2:] if len(contours) > 2 else [] 
    print(len(contours))
    cv2.drawContours(img2,    contours, -1, (255, 0, 0), -1)

    contours, hierarchy = cv2.findContours(thresh3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f'Number of contours: {len(contours)}')
    # contours = contours[2:] if len(contours) > 2 else [] 
    print(len(contours))
    cv2.drawContours(img3,    contours, -1, (255, 0, 0), -1)

    # Display
    titles = ['Blue - Global Threshold', 'Blue - Otsu', 'Blue - Otsu + Threshold', 
              'Global Threshold - Morphology', 'Otsu - Morphology', 'Otsu + Threshold - Morphology', 
              'Vegetable Outline', 'Vegetable Outline', 'Vegetable Outline', 
              'Vegetable Mask',  'Vegetable Mask',  'Vegetable Mask',  
              'Peeled Mask',  'Peeled Mask',  'Peeled Mask']  

    images = [b1, b2, b3, 
              clean1, clean2, clean3, 
              img1, img2, img3, 
              mask1, mask2, mask3, 
              thresh1, thresh2, thresh3]

    # Display image
    for i in range(15):
        plt.subplot(5,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



