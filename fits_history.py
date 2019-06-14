#!/usr/bin/env python

import argparse
from astropy.io import fits

parser = argparse.ArgumentParser()
parser.add_argument('-fits_file', dest='fits_file', help='fits_file')

args = parser.parse_args()
fits_file = args.fits_file

with fits.open(fits_file) as hdu:
    print(hdu[0].header['HISTORY'])
