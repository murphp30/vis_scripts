#!/usr/bin/env python
import subprocess
from astropy.io import fits
from astropy.coordinates import Angle
import astropy.units as u
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import glob
import numpy as np
import argparse
sys.path.append('/mnt/murphp30_data/typeIII_int/scripts')

from pearse_plot_fits import plot_burst
plt.ion()

def is_str_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    Taken from http://code.activestate.com/recipes/577058/
    
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def query_number_iterations(question, default=10):
    """
    Based off above function    
    """
    #if default is None:
    prompt = " [integer value] "
    #elif default == "yes":
    #    prompt = " [Y/n] "
    #elif default == "no":
    #    prompt = " [y/N] "
    #else:
    #    raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input()
        if default is not None and choice == '':
            return str(default)
        elif is_str_int(choice):
            return choice
        else:
            sys.stdout.write("Please respond with and integer\n")

def fits_data(fits_file):
    with fits.open(fits_file) as hdu:
        data = hdu[0].data[0,0]
    return data

def plot_all(niters):
    dirty = fits_data("wsclean-dirty.fits")
    image = fits_data("wsclean-image.fits")
    sun_rad = Angle(0.25*u.deg)
    with fits.open("wsclean-image.fits") as hdu:
        BMAJ = hdu[0].header["BMAJ"]
        BMIN = hdu[0].header["BMIN"]
        BPA = hdu[0].header["BPA"]
        head_str = fits.Header.tostring(hdu[0].header)
        str_start = head_str.find('scale')
        str_end = head_str.find('asec')
        axis_x = hdu[0].header["NAXIS1"]
        axis_y = hdu[0].header["NAXIS2"]
        scale = Angle(float(head_str[str_end-7:str_end])*u.arcsec)
        sun_x, sun_y = int(hdu[0].header['CRPIX1']), int(hdu[0].header['CRPIX2'])
        sun_x, sun_y = sun_x*scale.arcsec, sun_y*scale.arcsec
        MHz = hdu[0].header['CRVAL3']*1e-6
        obs_date = hdu[0].header["DATE-OBS"]    
    
    try:
        residuals = fits_data("wsclean-residual.fits")
    except FileNotFoundError:
        print("No residuals")
    psf = fits_data("wsclean-psf.fits")
    try:
        model = fits_data("wsclean-model.fits")
    except FileNotFoundError:
        print("No model")

    fig, axarr = plt.subplots(2,2, figsize=(8,8))
    im0 = axarr[0,0].imshow(image, aspect="equal", origin="lower",extent=[0,axis_x*scale.arcsec, 0, axis_y*scale.arcsec])
    b = Ellipse((1000,1000), Angle(BMAJ*u.deg).arcsec, Angle(BMIN*u.deg).arcsec,angle=90+BPA, fill=False, color='w')
    p = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
    
   # b = Ellipse((250,250), BMAJ/scale.deg, BMIN/scale.deg,angle=90+BPA, fill=False, color='w')
    axarr[0,0].set_title("Clean image " + niters + " iterations")
    axarr[0,0].add_patch(b)
    axarr[0,0].add_patch(p)
    fig.colorbar(im0, ax=axarr[0,0],fraction=0.046, pad=0.04)
    im1 = axarr[0,1].imshow(dirty, aspect="equal",origin="lower",extent=[0,axis_x*scale.arcsec, 0, axis_y*scale.arcsec])
    axarr[0,1].set_title("Dirty image " + niters + " iterations")
    p1 = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
    axarr[0,1].add_patch(p1)
    fig.colorbar(im1, ax=axarr[0,1],fraction=0.046, pad=0.04)
    
    if "residuals" in locals():
        im2 = axarr[1,0].imshow(residuals, aspect="equal",origin="lower",extent=[0,axis_x*scale.arcsec, 0, axis_y*scale.arcsec])
        axarr[1,0].set_title("Residuals " + niters + " iterations")
        p2 = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
        axarr[1,0].add_patch(p2)
        fig.colorbar(im2, ax=axarr[1,0],fraction=0.046, pad=0.04)
    if "model" in locals():
        im3 = axarr[1,1].imshow(model, aspect="equal",origin="lower",extent=[0,axis_x*scale.arcsec, 0, axis_y*scale.arcsec])
        axarr[1,1].set_title("Model " + niters + " iterations")
        p3 = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
        axarr[1,1].add_patch(p3)
        fig.colorbar(im3, ax=axarr[1,1],fraction=0.046, pad=0.04)
    plt.tight_layout()


parser = argparse.ArgumentParser()
parser.add_argument('niter_val',type=str, help='Number of iterations to run', default='1')
parser.add_argument('-m', '--model', dest='model', action='store_true',
                    help='Use to model on ../test_gauss-model.fits',default=False)

args = parser.parse_args()
model = args.model

set_params = "-j 36 -mem 85 -no-reorder -no-update-model-required -weight briggs -1 -mgain 0.85 -size 2048 2048 -scale 2.8752asec -pol I -data-column CORRECTED_DATA  -intervals-out 1 -interval 77194 77195 -use-differential-lofar-beam -fit-beam -make-psf"

j = "-j"
j_val = "36"
mem = "-mem"
mem_val = "85"
no_reorder = "-no-reorder"
no_update_model_required = "-no-update-model-required"
weight = "-weight"
briggs = "briggs"
briggs_val = "-1"
mgain = "-mgain"
mgain_val = "1"
size = "-size"
size_x = "1900"#"2048"
size_y = "1900"#"2048"
scale  = "-scale"
scale_val = '5.2635asec'#"2.8752asec"
pol = "-pol"
pol_val = "I"
data_column = "-data-column"
if model:
    data_column_val = "MODEL_DATA"
else:
    data_column_val = "CORRECTED_DATA"
intervals_out = "-intervals-out"
intervals_out_val = "1"
interval = "-interval"
interval_start = '77137'#"77127"#"57111"#"50330"#"76693"
interval_end =  '77138'#"77128" #"57112"#"51520" #"77892"
use_diff_lofar_beam = "-use-differential-lofar-beam"
multiscale = "-multiscale"
fit_beam = "-fit-beam"
make_psf = "-make-psf"
niter = "-niter"
niter_val = args.niter_val#sys.argv[1]  #try int except "please input integer"

mset = '/mnt/murphp30_data/typeIII_int/L401003_SB075_uv.dppp.flagMS'#"/mnt/murphp30_data/typeIII_int/L401003_SB064_uv.dppp.MS"
first_run = True
while True:
    #niter_param = "-niter {}".format(niters)
    #for i in range(len(f_list)):
    if first_run:
        #plot_burst(0, f_list).plot_fits(False)
        if model:
            subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
                no_update_model_required,
                size, size_x, size_y, scale, scale_val, pol, pol_val,
                interval, interval_start, interval_end, 
                fit_beam, make_psf,"-predict", "-name", "../test_gauss" , mset])
        subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
            no_update_model_required, weight, briggs, briggs_val, mgain,
            mgain_val, size, size_x, size_y, scale, scale_val, pol, pol_val,
            data_column, data_column_val,
            interval, interval_start, interval_end, use_diff_lofar_beam,
            fit_beam, make_psf, niter, niter_val, mset])
        plot_all(niter_val)
        #prev_data = plot_burst(0, f_list).data
        
        first_run = False
    
    elif not first_run:
        #fig, ax = plt.subplots(figsize=(6,6))
        #im = ax.imshow(prev_data, origin="lower", aspect="equal",vmax=np.percentile(prev_data, 99.9), vmin=np.percentile(prev_data, 0.1))
        #ax.set_title("Previous Run")
        #fig.colorbar(im, ax=ax)
        #plt.tight_layout()
        #plot_burst(0, f_list).plot_fits(False)
        #prev_data = plot_burst(0, f_list).data

        plot_all(str(int(niter_val) - int(increase)))
        if model:
            subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
                no_update_model_required,
                size, size_x, size_y, scale, scale_val, pol, pol_val,
                interval, interval_start, interval_end,
                fit_beam, make_psf,"-predict", "-name", "../test_gauss" , mset])
        subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
            no_update_model_required, weight, briggs, briggs_val, mgain,
            mgain_val, size, size_x, size_y, scale, scale_val, pol, pol_val,
            data_column, data_column_val, intervals_out, intervals_out_val,
            interval, interval_start, interval_end,use_diff_lofar_beam,
            fit_beam, make_psf,niter, niter_val, mset])
    
        plot_all(niter_val)    
    if query_yes_no("Increase iterations?"):
        plt.close("all")
        increase = query_number_iterations("Increase by how much?")
        niter_val = str(int(niter_val) + int(increase))
    else :
        plt.savefig("Summary_niter_{}.png".format(niter_val))
        plt.close("all")
        print("Finished at -niter {}".format(niter_val))
        #plot_burst(0, f_list).plot_fits()
        break


