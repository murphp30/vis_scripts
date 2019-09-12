#!/usr/bin/env python
#script to make movie from png files

import glob
import cv2
import os
import sys
import argparse
import pdb





parser = argparse.ArgumentParser()
parser.add_argument('-indir', dest='indir', help='input directory', default='./')
args = parser.parse_args()
indir = args.indir
#pdb.set_trace()

images = glob.glob(indir+'*.png')
images.sort()
output = indir+'sun.mp4'
image_path = images[0]
frame = cv2.imread(image_path)
cv2.imshow('video', frame)
height, width, channels = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter(output, fourcc, 10., (width, height))

for image in images:
     image_path = image
     frame = cv2.imread(image_path)
     out.write(frame)
     cv2.imshow('video', frame)
     if (cv2.waitKey(1) & 0xFF) == ord('q'):
         break

out.release()
cv2.destroyAllWindows()
