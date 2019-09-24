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
parser.add_argument('-f', dest='filename', help='file name to search for', default='*.png')
parser.add_argument('-o', dest='outfile', help='file name of output', default='movie.mp4')
args = parser.parse_args()
indir = args.indir
filename = args.filename
outfile = args.outfile
#pdb.set_trace()

images = glob.glob(indir+filename)
images.sort()
output = indir+outfile
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
