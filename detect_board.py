from __future__ import print_function
from nis import match
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Code for adaptive cutting board detection')
parser.add_argument('--query', help='Path to input image 1.', default='imgs/template.jpg')
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

# Initiate SIFT detector
sift = cv2.SIFT_create()

# Find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray1,None)
kp2, des2 = sift.detectAndCompute(gray2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

# Get matches
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
print(f"Matches: {len(matches)}")

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
goodMatches = []

# Ratio test as per Lowe's paper
ratioThresh = 1
for i,(m,n) in enumerate(matches):
    if m.distance < ratioThresh * n.distance:
        matchesMask[i]=[1,0]
        goodMatches.append(m)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

# -------- Draw bounding box ---------
import math

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

# Find top two matches
goodMatches = sorted(goodMatches, key = lambda x:x.distance)
goodMatches = goodMatches[:2]
print(f'Good Matches: {goodMatches}')

# Get angles between top two points in query and train images
queryPt1 = kp1[goodMatches[0].queryIdx].pt 
queryPt2 = kp1[goodMatches[1].queryIdx].pt
trainPt1 = kp2[goodMatches[0].trainIdx].pt
trainPt2 = kp2[goodMatches[1].trainIdx].pt
queryAngle =  np.arctan2(queryPt2[0] - queryPt1[0], queryPt2[1] - queryPt1[1])
print(f'Query Angle: {queryAngle}')
trainAngle =  np.arctan2(trainPt2[0] - trainPt1[0], trainPt2[1] - trainPt1[1])
print(f'Train Angle: {trainAngle}')
AngleDiff =  queryAngle - trainAngle

# Find center of query image in relation to top two query points
h, w = img1.shape[:2]
queryCenter = [w//2, h//2]
print(f"Height: {h} Width: {w} Center: {queryCenter}")

queryOffset = [queryCenter[0] - queryPt1[0] , queryCenter[1] - queryPt1[1]]
print(f'Query Offset: {queryOffset}')

# Find center in train image using top train point and angle difference
trainOffset = rotate([0,0], queryOffset, AngleDiff)
trainCenter = [int(trainPt1[0] + trainOffset[0]), int(trainPt1[1] + trainOffset[1])]
print(f"Train Center: {trainCenter}")

# Draw center
img3 = cv2.circle(img3, [trainCenter[0]  + w,    trainCenter[1]],  10, (255, 0, 0), -1)
img3 = cv2.circle(img3, [int(trainPt1[0] + w), int(trainPt1[1])],  5, (0, 0, 255), 2)
img3 = cv2.circle(img3, [int(trainPt2[0] + w), int(trainPt2[1])],  5, (0, 0, 0), 2)
img3 = cv2.circle(img3, queryCenter,                              10, (255, 0, 0), -1)
img3 = cv2.circle(img3, [int(queryPt1[0]),     int(queryPt1[1])],  5, (0, 0, 255), 2)
img3 = cv2.circle(img3, [int(queryPt2[0]),     int(queryPt2[1])],  5, (0, 0, 0), 2)

plt.imshow(img3,),plt.show()

 