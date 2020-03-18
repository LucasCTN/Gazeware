import cv2
import numpy as np
import math
import sys
import argparse


parser = argparse.ArgumentParser(description='Receives an image or video, and returns the position of the pupil. If empty, uses the webcam.')
parser.add_argument('--image_path', type=str, help='Path to an image file')
parser.add_argument('--video_path', type=str, help='Path to an video file')
args = parser.parse_args()

if args.image_path and args.video_path:
    parser.error("image_path and video_path can't be used at the same time.")

image_path = args.image_path
img = cv2.imread(image_path)
cv2.namedWindow('image')

def resize_image(img, pct):
    width = int(img.shape[1] * pct / 100)
    height = int(img.shape[0] * pct / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)

    return 4 * math.pi * area / perimeter**2

resized = resize_image(img, 100)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurried = cv2.medianBlur(gray, 5)

cv2.createTrackbar('thresh', 'image', 0, 255, lambda: None)

is_calibrating = True

thresh_value = 0
maxval = 255

while(True):
    if not is_calibrating:
    thresh_value = cv2.getTrackbarPos('thresh','image')

    new_blurried = blurried.copy()

    ret,thresh = cv2.threshold(blurried, thresh_value, maxval, 0)
    inv_thresh = cv2.bitwise_not(thresh)

    mser = cv2.MSER_create()
    regions = mser.detectRegions(inv_thresh)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    contours = hulls

    contours = [cnt for cnt in contours if circularity(cnt) > 0.9]
    contours.sort(key=circularity, reverse=True) # orders from most circular to least

     # finding center of contour
    if contours:
        is_calibrating = False
        cv2.setTrackbarPos('thresh', 'image', thresh_value)

        # compute the center of the contour
        cnt = contours[0] # selects the roundest contour

        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        
        # draw the center of the contour on the image
        cv2.circle(new_blurried, (cX, cY), 7, (10, 10, 255), -1)
    elif thresh_value < 255 and is_calibrating:
        thresh_value += 1
            continue

    cv2.drawContours(new_blurried, contours, -1, (0,0,255), 1)

    cv2.imshow('image', np.hstack([gray, inv_thresh, new_blurried]))

    k = cv2.waitKey(1) & 0xFF # Esc
    if k == 27:
        break

cv2.destroyAllWindows()
