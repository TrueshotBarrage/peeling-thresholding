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
    # h, w, c = img.shape
    # minW, maxW = int(w/10), int(9*w/10)
    # print(f"Min: {minW} Max: {maxW}")
    # img = img[:, minW:maxW]
    # print(img.shape)

    # Normalize image to min and max intensity of 0 and 255
    print(f"Original min: {np.min(img)} max: {np.max(img)}")
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

    """
    Threshold grayscale image using the following three methods:
        1. Binary thresholding with threshold = 127
        2. Otsu method
        3. Gaussian filtering + Otsu method

    Input: Grayscale image
    Output: Three binary images, respectively processed by the three methods 
        described above
    """
    def threshold(img):
        # global thresholding
        ret1,th1 = cv2.threshold(img, 127,255, cv2.THRESH_BINARY)
        # Otsu's thresholding
        ret2,th2 = cv2.threshold(img, 0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(img, (5,5),0)
        ret3,th3 = cv2.threshold(blur,0,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th1, th2, th3

    # Roughly segment vegetable in image using thresholding
    # Will refine later
    imgThresh1, imgThresh2, imgThresh3 = threshold(gray)
    redThresh1, redThresh2, redThresh3 = threshold(R)
    greenThresh1, greenThresh2, greenThresh3 = threshold(G)
    blueThresh1, blueThresh2, blueThresh3 = threshold(B)

    # Display
    titles = ['Original', 'Red', 'Green', 'Blue', 
              'Original - Gray', 'Red - Gray', 'Green - Gray', 'Blue - Gray',
              'Original - Global', 'Red - Global', 'Green - Global', 'Blue - Global', 
              'Original - Otsu', 'Red - Otsu', 'Green - Otsu', 'Blue - Otsu', 
              'Original - Otsu + Gauss', 'Red - Otsu + Gauss', 'Green - Otsu + Gauss', 'Blue - Otsu + Gauss',
              ]
    images = [img, red, green, blue, 
            gray, R, G, B, 
            imgThresh1, redThresh1, greenThresh1, blueThresh1, 
            imgThresh2, redThresh2, greenThresh2, blueThresh2, 
            imgThresh3, redThresh3, greenThresh3, blueThresh3,
            ]

    # Display image
    for i in range(20):
        plt.subplot(5,4,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

    ###### Segment #######
    # Refine previous segmentation to get outline of entire vegetable and 
    # separate segmentation of peeled area.
    # Use blue channel as it seems to provide most accurate segmentations.
    # Of the three methods, find the one where the masked area is smallest as to 
    # not include bleed.

    """
    Perform morphology to remove isolated black dots surrounding vegetable and 
        close white areas within the vegetable

    Input: Binary image
    Output: Binary image that has been closed and then opened.  Sorry for the 
        bad documentation lol
    """
    def morph(img):
        kernel = np.ones((5,5),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((100,100),np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return img

    # Use morphology to remove isolated black spots and remove white openings in vegetable
    morph1 = morph(blueThresh1)
    morph2 = morph(blueThresh2)
    morph3 = morph(blueThresh3)

    # Get copies to draw on
    imgDraw1, imgDraw2, imgDraw3 = np.copy(img), np.copy(img), np.copy(img)

    """
    Finds contours in imgBinary and draws outline of first contour on imgDraw
    in green and creates binary mask of first contour.  Both images get cropped
    size of first contour
    
    Input: 
        imgDraw - RGB image
        imgBinary - Binary image
    
    Output: 
        imgDraw - RGB image with outline of first contour found drawn in green.
            Cropped to size of first contour.
        mask - Binary image of first contour found filled with white
            Cropped to size of first contour.
    """
    def findDrawFirstContour(imgDraw, imgBinary):
        # Get mask
        contours, hierarchy = cv2.findContours(np.invert(imgBinary), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Number of contours: {len(contours)}')
        cv2.drawContours(imgDraw,    contours, 0, (0, 255, 0), 3)
        mask = np.zeros_like(imgDraw[:,:,0])
        mask =  cv2.drawContours(mask, contours, 0, (255), -1)
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(contours[0])
        # to save the images
        imgDraw = imgDraw[y:y+h,x:x+w]
        mask= mask[y:y+h,x:x+w]

        return mask, imgDraw

    # Get masks and draw vegetable outline
    vegMask1, imgDraw1 = findDrawFirstContour(imgDraw1, morph1)
    vegMask2, imgDraw2 = findDrawFirstContour(imgDraw2, morph2)
    vegMask3, imgDraw3 = findDrawFirstContour(imgDraw3, morph3)

    # Threshold different images usings the same method that the binary images
    # fed into findDrawFirstContour were used to find the contour drawn on them
    # Sorry if that's confusing!
    thresh1, _, _ = threshold(cv2.cvtColor(imgDraw1, cv2.COLOR_BGR2GRAY))
    _, thresh2, _ = threshold(cv2.cvtColor(imgDraw2, cv2.COLOR_BGR2GRAY))
    _, _, thresh3 = threshold(cv2.cvtColor(imgDraw3, cv2.COLOR_BGR2GRAY))

    # Get peel masks.  First erode the vegetable masks and then bitwise_and with
    # thersholded images found right above.  Erosion is necessary as we don't
    # want to include the contour of the vegetable which will be included in the 
    # thresholded images above since the images fed into threshold() include the
    # vegetable contours
    def findDrawOverlap(imgDraw, vegMask, imgThresh):
        kernel = np.ones((10,10),np.uint8)
        # Erode imgMask to eliminate outline of mask when we bitwise_and
        vegMask = cv2.erode(vegMask,kernel,iterations = 1)
        overlap = cv2.bitwise_and(vegMask, imgThresh)
        contours, hierarchy = cv2.findContours(overlap, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print(f'Number of contours: {len(contours)}')
        # contours = contours[2:] if len(contours) > 2 else [] 
        print(len(contours))
        cv2.drawContours(imgDraw,    contours, -1, (255, 0, 0), -1)
        return overlap

    peelMask1 = findDrawOverlap(imgDraw1, vegMask1, thresh1)
    peelMask2 = findDrawOverlap(imgDraw2, vegMask2, thresh2)
    peelMask3 = findDrawOverlap(imgDraw3, vegMask3, thresh3)

    # Display
    titles = ['Blue - Global Threshold', 'Blue - Otsu', 'Blue - Otsu + Threshold', 
              'Global Threshold - Morphology', 'Otsu - Morphology', 'Otsu + Threshold - Morphology', 
              'Vegetable Outline', 'Vegetable Outline', 'Vegetable Outline', 
              'Vegetable Mask',  'Vegetable Mask',  'Vegetable Mask',  
              'Peeled Mask',  'Peeled Mask',  'Peeled Mask']  

    images = [blueThresh1, blueThresh2, blueThresh3, 
              morph1, morph2, morph3, 
              imgDraw1, imgDraw2, imgDraw3, 
              vegMask1, vegMask2, vegMask3, 
              peelMask1, peelMask2, peelMask3]

    # Display image
    for i in range(15):
        plt.subplot(5,3,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



