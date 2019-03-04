#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 12:49:31 2018
##written specifically for a FOV matching with LASCO-C2
@author: anshu
"""
import glob
import os
from astropy.io import fits
import cv2
import argparse
import matplotlib.pyplot as plt
import time
import numpy as np
from matplotlib import dates
import Tkinter
import tkFileDialog
import sunpy.sun
from astropy.coordinates import SkyCoord
from functools import wraps
from astropy.table import Table, Column
import h5py
import matplotlib
import datetime
import pylab 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import date2num
from matplotlib.mlab import griddata as gd
import sys
from matplotlib import dates
from pylab import figure,imshow,xlabel,ylabel,title,close,colorbar
from astropy.visualization import astropy_mpl_style
from astropy import units as u
import PIL
import pdb
'''
import Tkinter,tkFileDialog
root = Tkinter.Tk()
root.withdraw()
filename = tkFileDialog.askopenfile(parent=root,mode='r',title='Choose a fits file')
if filename != None:
    print "You chose %s", file


#filename1=tkFileDialog.askopenfilename()
#hdu1 = fits.open(filename1)[0]
#data1=hdu1.data
#wcs1 = WCS(hdu1.header)



#d=-23.45*np.sin(np.deg2rad(360.0/365.*(89.-81.)))
'''


font = {'family': 'times',
        'color':  'darkred',
        'weight': 'bold',
        'size': 16,
        }

###################################################################
#selcting the directory with fits files
root = Tkinter.Tk()
root.withdraw() #use to hide tkinter window
currdir = os.getcwd()
dir = tkFileDialog.askdirectory(parent=root, initialdir=currdir, title='Please select a directory')
if len(dir) > 0:
    print "You chose %s" % dir
    
####################################################################
start_time = time.time() #to display the execution time

#dir = "/home/anshu/Desktop/images_quiet_B/" 
#Enter the firt file name, just for taking time information
#hdulist = fits.open(os.path.join(dir,'20MHz-t0007.fits'))
hdulist = fits.open(os.path.join(dir,'wsclean-t0000-image.fits'))
hdu = hdulist[0]
a=hdu.header[43];
data=np.zeros((hdu.header[3],hdu.header[4]), dtype=int)

[RA0,DEC0]=sunpy.sun.position(t=sunpy.time.parse_time(hdu.header[43]))
c0 = SkyCoord(RA0,DEC0)
RA0=c0.ra.rad*57.3
DEC0=c0.dec.rad*57.3
RA0_a=np.linspace((RA0-.75),(RA0+.75),1024)
DEC0_a=np.linspace((DEC0-.75),(DEC0+.75),1024)

#####################################################################
#Reading all the FITS files in a list
files = list(glob.glob(os.path.join(dir,'*.fits')))
files = sorted(files, key = lambda x: x.rsplit('.', 1)[0]) # sort files in numerical order
print len(files)
#####################################################################
#plotting fits files and saving the figures
#plt.figure()
for i in range(len(files)):
    hdulist = fits.open(files[i])
    fits.info(files[i])
    print files[i]
    hdu = hdulist[0]
    a=hdu.header[43];
    freq = hdu.header[34]/1000000
    print a
    data = hdu.data
    data=data[0,0,:,:]
    print np.max(data)
   # pdb.set_trace()
    # correction for RA DEC because of movement of the Sun
    [RA,DEC]=sunpy.sun.position(t=sunpy.time.parse_time(hdu.header[43]))
    #[RA,DEC]=sunpy.sun.position(t='now')
    #[RA,DEC]=sunpy.sun.position(t='2014-02-27T00:04:18.298')
    c = SkyCoord(RA,DEC)
    c0 = SkyCoord(RA0,DEC, unit='deg')
    RA=c.ra.rad*57.3
    DEC=c.dec.rad*57.3
    data = hdu.data
    data=data[0,0,:,:]
    RA0=c0.ra.rad*57.3
    DEC0=c0.dec.rad*57.3
    RA0_a=np.linspace((RA0-.75),(RA0+.75),1024)
    DEC0_a=np.linspace((DEC0-.75),(DEC0+.75),1024)
    RA_a=np.linspace((RA-.75),(RA+.75),1024)
    DEC_a=np.linspace((DEC-.75),(DEC+.75),1024)
    x_sh=RA-RA0
    y_sh=DEC-DEC0
    one_pixel_x=abs(RA0_a[1]-RA0_a[2])
    one_pixel_y=abs(DEC0_a[1]-DEC0_a[2])
    one_deg_x=1.0/one_pixel_x
    one_deg_y=1.0/one_pixel_y
    shift_x =int( x_sh * one_deg_x)
    shift_y = int(y_sh * one_deg_y)
    data1=data[shift_x:len(data-1),shift_y:len(data)-1]
    pad=np.zeros((len(data),len(data)), dtype=int)
    pad[len(pad)-data1.shape[0]:len(pad),len(pad)-data1.shape[1]:len(pad)]=data1
    
    x=np.linspace(-512,512,1024)
    y=np.linspace(-512,512,1024)
    #plt.imshow(pad,aspect = 'auto',interpolation='nearest',cmap='Greys',extent=(min(RA0_a),max(RA0_a),min(DEC0_a),max(DEC0_a)))
    #plt.imshow(data,aspect = 'auto',interpolation='nearest', vmin=np.amax(data)*50/100,vmax=np.amax(data)*90/100,cmap='Greys',extent=(min(RA0_a),max(RA0_a),min(DEC0_a),max(DEC0_a)))
    #plt.imshow(data,aspect = 'auto',interpolation='nearest', vmin=np.amax(data)*50/100,vmax=np.amax(data)*90/100,cmap='Greys')
    #plt.imshow(pad,aspect = 'auto', vmin=0, vmax=400, interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))
    # plt.imshow(pad,aspect = 'auto', vmin=np.amax(data)*85/100,vmax=np.amax(data)*99/100 , interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))
    plt.imshow(pad,aspect = 'auto', vmin=np.amax(data)*5/100,vmax=np.amax(data)*99/100 , interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))
    #plt.imshow(pad,aspect = 'auto', interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))
    #plt.imshow(pad,aspect = 'auto',vmin=np.percentile(pad,98),vmax=np.percentile(pad,99.5), interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))
    #plt.imshow(pad,aspect = 'auto', interpolation='nearest',cmap='Greys',extent=(min(x),max(x),min(y),max(y)))

    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=14)
    plt.xlabel('RA (arcsec)',fontsize=14)
    plt.ylabel('DEC (arcsec)',fontsize=14)
    plt.title(hdu.header[43])
    plt.colorbar()
    
    #plt.text(RA0_a[700],DEC0_a[100], str(int(hdu.header[34]/1000000)) + 'MHz', fontdict=font)
    plt.text(x[700],y[100], str(int(hdu.header[34]/1000000)) + 'MHz', fontdict=font)
    #plt.text(x[800],y[950], 'file:'+str(i), fontdict=font)
    ##############################################################
    #Solar disk!!! 
    #circle1 = plt.Circle((RA0_a[511],DEC0_a[511]), DEC0_a[511+85]-DEC0_a[511], color='r',fill=False)
    circle1 = plt.Circle((x[511],y[511]), x[511+85]-x[511], color='r',fill=False)
    pdb.set_trace()
    fig = plt.gcf()
    ax = fig.gca()
    ax.add_artist(circle1)
    ax.set_aspect('equal')
    plt.savefig(os.path.join(dir, str(i).rjust(3,'0')+'.png'))
    print (str(i) + ".png")
    plt.clf()
    #close()
    #plt.show()
######################################################################
#making video fromt the saved images in previous step
    #(next lines are to modify the code and pass the arguments)
ap = argparse.ArgumentParser()
ap.add_argument("-ext", "--extension", required=False, default='png', help="extension name. default is 'png'.")
ap.add_argument("-o", "--output", required=False, default='output.mp4', help="output video file")
args = vars(ap.parse_args())
ext = args['extension']
output = args['output']
images = []
for f in os.listdir(dir):
    if f.endswith(ext):
        images.append(f)
        
# Determine the width and height from the first image
image_path = os.path.join(dir, images[0])
frame = cv2.imread(image_path)
cv2.imshow('video',frame)
height, width, channels = frame.shape

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
out = cv2.VideoWriter(os.path.join(dir,output), fourcc, 20.0, (width, height))
#sorting the list numerically
images = sorted(images, key = lambda x: x.rsplit('.', 1)[0]) 

for image in images:
    image_path = os.path.join(dir, image)
    frame = cv2.imread(image_path)
    out.write(frame) # Write out frame to video
    cv2.imshow('video',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'): # Hit `q` to exit
        break
# Release everything if job is finished
out.release()
print "You chose %s" % dir
cv2.destroyAllWindows()
close()
# done done done
print("The output video is {}".format(output))
print dir
print("--- %s seconds ---" % (time.time() - start_time))
