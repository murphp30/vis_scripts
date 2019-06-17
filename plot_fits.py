#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse,Circle
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from astropy.coordinates import SkyCoord, Angle, EarthLocation, AltAz
import astropy.units as u
import sys
import pdb
import scipy.optimize as opt
import sunpy
import sunpy.map
from sunpy.map import header_helper
import sunpy.coordinates
import sunpy.coordinates.sun as sun
from sunpy.coordinates import frames




#prefix = sys.argv[3]
def fits_data(fits_file):
    with fits.open(fits_file) as hdu:
        data = hdu[0].data[0,0]
    return data

def poly_area(vs):
    s = []
    for i in range(len(vs)-1):
        xi = vs[i][0]
        xi1 = vs[i+1][0]
        yi = vs[i][1]
        yi1 = vs[i+1][1]
        s.append((xi*yi1-xi1*yi))
    return 0.5*np.sum(s)

def LOFAR_to_sun(smap):
    obs_coord = SkyCoord(smap.reference_coordinate, distance=sun.earth_distance(smap.date),obstime=smap.date)
    rot_ang = sun.P(smap.date) 
    smap_rot = smap.rotate(angle=-rot_ang)
    smap_out_head = header_helper.make_fitswcs_header(smap_rot.data,obs_coord.transform_to(frames.Helioprojective),
            u.Quantity(smap_rot.reference_pixel),u.Quantity(smap_rot.scale))
    smap_out = sunpy.map.Map(smap_rot.data, smap_out_head)
    smap_out.meta['wavelnth'] = smap.meta['crval3']/1e6
    smap_out.meta['waveunit'] = "MHz"
    return smap_out

def plot_all(niters):
    dirty = fits_data(prefix+"-dirty.fits")
    image = fits_data(prefix+"-image.fits")
    with fits.open(prefix+"-image.fits") as hdu:
        BMAJ = hdu[0].header["BMAJ"]
        BMIN = hdu[0].header["BMIN"]
        BPA = hdu[0].header["BPA"]
        head_str = fits.Header.tostring(hdu[0].header)
        str_start = head_str.find('scale')
        str_end = head_str.find('asec')
        scale = Angle(float(head_str[str_end-7:str_end])*u.arcsec)
        sun_x, sun_y = int(hdu[0].header['CRPIX1']), int(hdu[0].header['CRPIX2'])
    sun_rad = Angle(0.25*u.deg)
    sun_rad_pix = sun_rad.deg/scale.deg
    try:
        residuals = fits_data(prefix+"-residual.fits")
    except FileNotFoundError:
        print("No residuals")
    psf = fits_data(prefix+"-psf.fits")
    try:
        model = fits_data(prefix+"-model.fits")
    except FileNotFoundError:
        print("No model")

    fig, axarr = plt.subplots(2,2, figsize=(8,8))
    im0 = axarr[0,0].imshow(image, aspect="equal", origin="lower")
    b = Ellipse((250,250), BMAJ/scale.deg, BMIN/scale.deg,angle=90+BPA, fill=False, color='w') 
    p = Circle((sun_x, sun_y),sun_rad_pix, fill=False, color='r')
    axarr[0,0].add_patch(b)
    axarr[0,0].add_patch(p)
    axarr[0,0].set_title("Clean image " + niters + " iterations")
    fig.colorbar(im0, ax=axarr[0,0])
    im1 = axarr[0,1].imshow(dirty, aspect="equal",origin="lower")
    axarr[0,1].set_title("Dirty image " + niters + " iterations")
    fig.colorbar(im1, ax=axarr[0,1])
    
    if "residuals" in locals():
        im2 = axarr[1,0].imshow(residuals, aspect="equal",origin="lower")
        axarr[1,0].set_title("Residuals " + niters + " iterations")
        fig.colorbar(im2, ax=axarr[1,0])
    if "model" in locals():
        im3 = axarr[1,1].imshow(model, aspect="equal",origin="lower")
        axarr[1,1].set_title("Model " + niters + " iterations")
        fig.colorbar(im3, ax=axarr[1,1])
    plt.tight_layout()

def plot_one(fits_in,gfit=False):
    data = fits_data(fits_in)
    with fits.open(fits_in) as hdu:
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
    sun_rad = Angle(0.25*u.deg)
    #sun_rad_pix = sun_rad.deg/scale.deg
    fig, ax = plt.subplots()
    im=ax.imshow(data, aspect="equal", origin="lower", extent=[0,axis_x*scale.arcsec, 0, axis_y*scale.arcsec], vmin=np.percentile(data,0.1), vmax=np.percentile(data, 99.9))
#np.percentile(data, .1),
    #b = Ellipse((250,250), BMAJ/scale.deg, BMIN/scale.deg,angle=90+BPA, fill=False, color='w')
    if fits_in[-8:] != "psf.fits":
        b = Ellipse((1000,1000), Angle(BMAJ*u.deg).arcsec, Angle(BMIN*u.deg).arcsec,angle=90+BPA, fill=False, color='w',ls='--')
        p = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
        ax.add_patch(b)
        ax.add_patch(p)
    else:
    
        b = Ellipse((sun_x,sun_y), Angle(BMAJ*u.deg).arcsec, Angle(BMIN*u.deg).arcsec,angle=90+BPA, fill=False, color='w')
        #p = Circle((sun_x, sun_y),sun_rad.arcsec, fill=False, color='r')
        ax.add_patch(b)
        #ax.add_patch(p)
    
    if gfit:
        #popt = gauss_params(data,axis_x)[0]
        #g_x,g_y = popt[1]*scale, popt[2]*scale
        #fwhmx = 2*np.sqrt(2*np.log(2))*popt[3]*scale
        #fwhmy = 2*np.sqrt(2*np.log(2))*popt[4]*scale
        #rot = np.mod(Angle(popt[5]*u.rad).deg,360)
        #g = Ellipse((g_x.arcsec,g_y.arcsec), fwhmx.arcsec, fwhmy.arcsec,angle=-rot, fill=False, color='w')
        #ax.add_patch(g)
        #pdb.set_trace() 
        X = np.arange(axis_x)*scale.arcsec
        Y = np.arange(axis_y)*scale.arcsec
        X,Y = np.meshgrid(X,Y)
        ax.contour(X,Y,data,[.5*np.max(data), .9*np.max(data), .97*np.max(data)],colors='w')
        
        #pdb.set_trace()
    fig.colorbar(im, ax=ax)
    plt.title(str(np.round(MHz,3)) + " MHz "+ obs_date + " ")
    plt.xlabel('arcsec')
    plt.ylabel('arcsec')
#    plt.savefig(out_png)
def fit_2d_gauss(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
    #theta measured clockwise! N.B theta is ANTIclockwise in matplotlib
        (x, y) = xy
        x0 = float(x0)
        y0 = float(y0)
        a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
        b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
        c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
        g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
        return g.ravel()

def gauss_params(data,pix_len):
    #gauss parameters in pixel space
        x = np.arange(pix_len)
        y = np.arange(pix_len)
        max_x, max_y = np.where(data == np.max(data))
        max_x, max_y = x[max_x[0]], y[max_y[0]]
        guess = (np.max(data),max_x, max_y, 10, 10, 0, 0)
        bnds = ([0,0,0,0,0,-np.inf, -np.inf],[np.inf,x[-1],y[-1],np.inf, np.inf,np.inf,np.inf])
        x, y = np.meshgrid(x,y)
        try:
            popt, pcov = opt.curve_fit(fit_2d_gauss, (x,y), data.ravel(), p0=guess, bounds=bnds)
        except RuntimeError:
            return []
        return popt, pcov


if __name__ == "__main__":

    fits_in = sys.argv[1]
    out_png = sys.argv[2]
    # plot_one(fits_in)#, True)    
    # plt.savefig(out_png)
#plot_all('1')
    #plt.show()
    # data = fits_data(fits_in)
    # with fits.open(fits_in) as hdu:
    #     BMAJ = hdu[0].header["BMAJ"]
    #     BMIN = hdu[0].header["BMIN"]
    #     BPA = hdu[0].header["BPA"]
    #     head_str = fits.Header.tostring(hdu[0].header)
    #     str_start = head_str.find('scale')
    #     str_end = head_str.find('asec')
    #     axis_x = hdu[0].header["NAXIS1"]
    #     axis_y = hdu[0].header["NAXIS2"]
    #     scale = Angle(float(head_str[str_end-7:str_end])*u.arcsec)
    #     sun_x, sun_y = int(hdu[0].header['CRPIX1']), int(hdu[0].header['CRPIX2'])
    #     sun_x, sun_y = sun_x*scale.arcsec, sun_y*scale.arcsec
    #     MHz = hdu[0].header['CRVAL3']*1e-6
    #     obs_date = hdu[0].header["DATE-OBS"]
    # sun_rad = Angle(0.25*u.deg)

    # LOFAR_centre = [3826577.066*u.m, 461022.948*u.m, 5064892.786*u.m]
    # LOFAR_earth = EarthLocation(*LOFAR_centre)
    # wcs = WCS(fits_in)
    smap = sunpy.map.Map(fits_in)
    smap.meta['wavelnth'] = smap.meta['crval3']/1e6
    smap.meta['waveunit'] = "MHz"
    # smap.plot(cmap='viridis')
    # plt.figure()
    helio_smap = LOFAR_to_sun(smap)
    helio_smap.plot(cmap='viridis')
    helio_smap.draw_limb(color='r')
    plt.savefig(out_png)
    
    #pdb.set_trace()
    """
    popt = gauss_params(data,axis_x)
    g_x,g_y = popt[1]*scale, popt[2]*scale
    fwhmx = 2*np.sqrt(2*np.log(2))*popt[3]*scale
    fwhmy = 2*np.sqrt(2*np.log(2))*popt[4]*scale
    rot = np.mod(Angle(popt[5]*u.rad).deg,360)

    params = np.array([g_x,g_y,fwhmx, fwhmy, rot])
    np.save(fits_in[:-4], params)
    """
