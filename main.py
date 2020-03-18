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

resized = resize_image(img, 100)
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
blurried = cv2.medianBlur(gray, 5)

while(True):
    cv2.imshow('image', gray)

    k = cv2.waitKey(1) & 0xFF # Esc
    if k == 27:
        break

cv2.destroyAllWindows()
