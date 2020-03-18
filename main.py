import cv2
import numpy as np
import math
import sys
import argparse


THRESHOLD_SEARCH_STEP = 5
CIRCULARITY_TOLERANCE = 0.8

parser = argparse.ArgumentParser(description='Receives an image or video, and returns the position of the pupil.')
parser.add_argument('--image_path', type=str, help='Path to an image file')
parser.add_argument('--video_path', type=str, help='Path to an video file or 0 if it\'s a webcam.')
parser.add_argument('--flip', help='Image/video should flip vertically.')
parser.add_argument('--no_loop', help='Video should not loop when it ends.')
args = parser.parse_args()

if args.image_path and args.video_path:
    parser.error("image_path and video_path can't be used at the same time.")

if args.image_path:
    image_path = args.image_path
    img = cv2.imread(image_path)
elif args.video_path:
    video_path = args.video_path
    if video_path.isnumeric():
        video_path = int(video_path)
    cap = cv2.VideoCapture(video_path)

cv2.namedWindow('image')

def resize_image(img):
    max_height = 400
    max_width = 400

    width = int(img.shape[1])
    height = int(img.shape[0])

    aspect_ratio = width / height

    dim = (width, height)
    
    if height > max_height:
        new_heigth = 400
        new_width = int(new_heigth * aspect_ratio)
        dim = (new_width, new_heigth)
    
    if width > max_width:
        new_width = 400
        new_heigth = int(new_width / aspect_ratio)
        dim = (new_width, new_heigth)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour,True)
    return 4 * math.pi * area / perimeter**2

if args.image_path:
    resized = resize_image(img)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurried = cv2.medianBlur(gray, 5)

cv2.createTrackbar('thresh', 'image', 0, 255, lambda: None)

is_calibrating = True

thresh_value = 0
maxval = 255

while(True):
    if not is_calibrating:
        thresh_value = cv2.getTrackbarPos('thresh','image')
    
    if args.video_path:
        ret, frame = cap.read()
        if ret:
            if args.flip:
                frame = cv2.flip(frame, flipCode=-1)
            resized = resize_image(frame)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            blurried = cv2.medianBlur(gray, 5)
        elif not args.no_loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    new_blurried = blurried.copy()
    new_img = resized.copy()

    ret,thresh = cv2.threshold(blurried, thresh_value, maxval, 0)
    inv_thresh = cv2.bitwise_not(thresh)

    mser = cv2.MSER_create()
    regions = mser.detectRegions(inv_thresh)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]
    contours = hulls

    contours = [cnt for cnt in contours if circularity(cnt) > CIRCULARITY_TOLERANCE]
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
        cv2.circle(new_img, (cX, cY), 7, (10, 10, 255), -1)
    elif thresh_value < 255 and is_calibrating:
        thresh_value += THRESHOLD_SEARCH_STEP
        continue

    cv2.drawContours(new_img, contours, -1, (0,0,255), 1)

    bgr_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    bgr_inv_thresh = cv2.cvtColor(inv_thresh, cv2.COLOR_GRAY2BGR)
    cv2.imshow('image', np.hstack([bgr_gray, bgr_inv_thresh, new_img]))

    k = cv2.waitKey(1) & 0xFF # Esc
    if k == 27:
        break

cv2.destroyAllWindows()
