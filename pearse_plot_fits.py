#!/usr/bin/env python
#script to view fits files generated by wsclean

import glob
import h5py
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.patches import Circle, Ellipse
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta
from astropy.io import fits
import cv2
import os
import numpy as np
import sunpy.sun
import sunpy.coordinates
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord, Angle, get_sun
from astropy.table import Table, Column
from astropy.constants import c, e, m_e, eps0, R_sun
import astropy.time
import sys
from matplotlib import dates
from astropy.visualization import astropy_mpl_style
from astropy import units as u
import pdb
import scipy.optimize as opt
from multiprocessing import Pool
import argparse
#assume all images are made with sam escale size etc

#dec_scale = Angle(np.zeros(no_pix_y), unit='deg')
#dec_scale[513] = get_grid().dec
#dec_scale = [dec_scale[513] + (i-513)*scale for i in range(no_pix_y)]
#
#ra_scale = Angle(np.zeros(no_pix_x), unit='hourangle')
#ra_scale[513] = get_grid().ra
#ra_scale = [ra_scale[513] + (i-513)*scale for i in range(no_pix_x)]
#f_list=["*.fits"]
class plot_burst:
        """
        class to define grid for correct coordinates
        and plot interferometric image
        files = list of fits files
        xpix, ypix = no of x,y pixels
        t = image index
        """
        def __init__(self,t):#,files=f_list):
                self.files = f_list#files
                self.t = t

                with fits.open(self.files[self.t]) as hdu:
                        head_str = fits.Header.tostring(hdu[0].header)
                        str_start = head_str.find('scale')
                        str_end = head_str.find('asec')
                        self.scale = Angle(float(head_str[str_end-7:str_end])*u.arcsec)
                        
                        self.xpix = hdu[0].header["NAXIS1"]
                        self.ypix = hdu[0].header["NAXIS2"]

                        self.data = hdu[0].data[0,0]
                        self.obs_time = hdu[0].header['DATE-OBS']
                        self.sun_pos = get_sun(astropy.time.Time(self.obs_time))#sunpy.sun.position(t=sunpy.time.parse_time(obs_time)) 
                        self.BMAJ = hdu[0].header['BMAJ']
                        self.BMIN = hdu[0].header['BMIN']
                        self.BPA = hdu[0].header['BPA']
                        self.MHz = hdu[0].header['CRVAL3']*1e-6
                        self.ra_centre = int(hdu[0].header['CRPIX1']) 
                        self.dec_centre = int(hdu[0].header['CRPIX2'])
        
                        self.obs_ra = hdu[0].header['CRVAL1']
                        self.obs_dec = hdu[0].header['CRVAL2']
        
#        def bf_time_slice(self):
#            
#            bf, freq, t_arr = get_data(bf_file)
            


        def get_grid(self):
           
                #obs_centre = obs_time[:11]+'12:00:00.0'
                obs_ra = Angle(self.obs_ra, unit='deg').wrap_at('360d')
                obs_dec = Angle(self.obs_dec, unit='deg')
                #RA, DEC = sunpy.sun.position(t=sunpy.time.parse_time(obs_centre))
                DIST = sunpy.coordinates.get_sunearth_distance(sunpy.time.parse_time(self.obs_time))
                grid_eq = SkyCoord(obs_ra, obs_dec, DIST, obstime=self.obs_time)
                grid_helio = grid_eq.transform_to(frames.Helioprojective)
                return grid_eq
        
        def find_sun(self):
                #find centre of sun at time t in some coordinate system
                RA, DEC = sunpy.sun.position(t=sunpy.time.parse_time(self.obs_time))
                DIST = sunpy.coordinates.get_sunearth_distance(sunpy.time.parse_time(self.obs_time)) 
                sun_eq = SkyCoord(RA, DEC, DIST, obstime=self.obs_time)
                sun_helio = sun_eq.transform_to(frames.Helioprojective)
                return sun_helio
        
        def gauss_params(self): 
                x = np.arange(self.xpix)
                y = np.arange(self.ypix)
                x, y = np.meshgrid(x,y)
                max_x, max_y = np.where(self.data == np.max(self.data))
                max_x, max_y = max_x[0], max_y[0]
                guess = (np.max(self.data),max_x, max_y, 0.2, 0.2, 0, 0)
                try:
                    popt, pcov = opt.curve_fit(fit_2d_gauss, (x,y), self.data.ravel(), p0=guess)
                except RuntimeError:
                    return [] 
                return popt, pcov

        def plot_sun_pix(self):
        #plot image with sun overlaid, also include beam        
                fig,(ax,ax1,ax2) = plt.subplots(3,1,figsize=(7,7), gridspec_kw={'height_ratios':[3,1,1]})
                ax.imshow(self.data, origin="lower", aspect="equal",vmax=np.percentile(self.data, 99.9), vmin=np.percentile(self.data, 0.1))
                sun_x, sun_y = self.ra_centre, self.dec_centre#coord2pix(sun_pos.ra, sun_pos.dec)
                ax.scatter(sun_x, sun_y, c='r', marker='+')
                
#                I_data, freq, t_arr = get_data(bf_file)
                x = np.arange(self.xpix)
                y = np.arange(self.ypix)
                x, y = np.meshgrid(x,y)
                
                start_time = get_obs_start(bf_file)
                start_time = astropy.time.Time(start_time).datetime
                cusum_start = self.get_burst_start()
                cusum_end = self.get_burst_start(reverse=True)
                burst_start = start_time + timedelta(seconds = t_arr[cusum_start])
                burst_end = start_time + timedelta(seconds = t_arr[cusum_end])
                obs = astropy.time.Time(self.obs_time).datetime
                #if (obs >= burst_start) and (obs <= burst_end):
                #        popts, pcovs = self.gauss_params()      
                #        gss = fit_2d_gauss((x,y),popts[0], popts[1], popts[2], popts[3], popts[4], popts[5], popts[6]).reshape(self.xpix,self.ypix)
                #        ax.contour(gss, [0.5*gss.max()],colors='r')
                #        ax.scatter(popts[1], popts[2], c='r', marker='+')
                
                #sun_max = np.where(data==np.max(data))
                #ax.scatter(ra_scale[sun_max[1][0]].deg, dec_scale[sun_max[0][0]].deg, c='r', marker='+') #these are reversed to what you would think
                sun_rad = Angle(0.25*u.deg)
                sun_rad_pix = sun_rad.deg/self.scale.deg
                p = Circle((sun_x, sun_y),sun_rad_pix, fill=False, color='r')
                b = Ellipse((100,100),self.BMAJ/self.scale.deg,self.BMIN/self.scale.deg,angle=90+self.BPA, fill=False, color='w') 
           # ax = plt.gca()
                ax.add_patch(p)
                ax.add_patch(b)
                ax.set_xlabel('RA (pix)')
                ax.set_ylabel('Declination (pix)')
                ax.set_title("Sun at "+"{:.2f}".format(self.MHz)+ " MHz "+self.obs_time)
                
                start_time = get_obs_start(bf_file)
                percentile_min = .1
                percentile_max = 99.9
                dt = t_arr[1]-t_arr[0]
                bf_MHz =  np.where(np.isclose(freq, self.MHz, atol=1e-2)==True)[0][0]
                datetime_start = datetime.strptime(start_time.decode("utf-8")[:-4], "%Y-%m-%dT%H:%M:%S.%f")
                timedelta_zoom_start = timedelta(seconds=t_arr[0])
                datetime_zoom_start = datetime_start + timedelta_zoom_start
                datetime_zoom_end = datetime_start + timedelta(seconds=t_arr[-1])
                tarrdelta = t_arr*timedelta(seconds=1)
                taxis = tarrdelta + datetime_start
                xlims = list([datetime_zoom_start, datetime_zoom_end])
                xlims = dates.date2num(xlims)
                
                bg_sub = I_data/np.mean(I_data, axis=0)
                im = ax1.imshow(bg_sub.T, aspect="auto", extent=[xlims[0], xlims[-1], freq[-1], freq[0]],
                                        vmax=np.percentile(bg_sub.T, percentile_max), vmin=np.percentile(bg_sub.T, percentile_min))
        
                
                ax1.xaxis_date()
                date_format = dates.DateFormatter("%H:%M:%S")
                ax1.xaxis.set_major_formatter(date_format)
                ax1.axvline(x=dates.date2num(astropy.time.Time(self.obs_time).datetime), color='r')
                ax1.axhline(y=self.MHz, color='r')      
                #ax1.set_xlabel("Time (UTC) ")
                ax1.set_ylabel("Frequency (MHz)")
                ax1.get_shared_x_axes().join(ax1, ax2)
                ax1.set_xticklabels([])
                ax2.plot(taxis, bg_sub[:, bf_MHz])
                #ax2.xaxis_date()
                #ax2.xaxis.set_major_formatter(date_format)
                ax2.axvline(x=dates.date2num(astropy.time.Time(self.obs_time).datetime), color='r')
                ax2.set_xlabel("Time (UTC) ")
                ax2.set_ylabel("Intensity (AU)")
                plt.tight_layout()
                plt.savefig(indir + str(self.t).zfill(4) +'.png')
                plt.close()
        #plt.show()
        
        def plot_just_sun_pix(self, save=True):
        #plot image with sun overlaid, also include beam        
                fig,ax = plt.subplots(figsize=(7,7))
                im = ax.imshow(self.data, origin="lower", aspect="equal",vmax=np.percentile(self.data, 99.9),vmin=np.percentile(self.data, 0.1))
                sun_x, sun_y = self.ra_centre, self.dec_centre#coord2pix(sun_pos.ra, sun_pos.dec)
                ax.scatter(sun_x, sun_y, c='r', marker='+')
                sun_rad = Angle(0.25*u.deg)
                sun_rad_pix = sun_rad.deg/self.scale.deg
                p = Circle((sun_x, sun_y),sun_rad_pix, fill=False, color='r')
                b = Ellipse((100,100),self.BMAJ/self.scale.deg,self.BMIN/self.scale.deg,angle=90+self.BPA, fill=False, color='w') 
                ax.add_patch(p)
                ax.add_patch(b)
                ax.set_xlabel('RA (pix)')
                ax.set_ylabel('Declination (pix)')
                ax.set_title("Sun at "+"{:.2f}".format(self.MHz)+ " MHz "+self.obs_time)
                fig.colorbar(im, ax=ax,fraction=0.046, pad=0.04) 
                plt.tight_layout()
                if save:
                    plt.savefig(indir + str(self.t).zfill(4) +'sun_only.png')
                    plt.close()
                else:
                    plt.show()
        
        def plot_fits(self, save=True):
        #plot image with sun overlaid, also include beam        
                fig,ax = plt.subplots(figsize=(6,6))
                im = ax.imshow(self.data, origin="lower", aspect="equal",vmax=np.percentile(self.data, 99.9), vmin=np.percentile(self.data, 0.1))
                b = Ellipse((100,100),self.BMAJ/self.scale.deg,self.BMIN/self.scale.deg,angle=90-self.BPA+23.5, fill=False, color='w') 
                ax.add_patch(b)
                ax.set_xlabel('RA (pix)')
                ax.set_ylabel('Declination (pix)')
                ax.set_title("Sun at "+"{:.2f}".format(self.MHz)+ " MHz "+self.obs_time + " " + self.files[self.t][7:-5])
                fig.colorbar(im, ax=ax) 
                plt.tight_layout()
                if save:
                    plt.savefig(indir + str(self.t).zfill(4) +self.files[self.t][7:-5] +'.png')
                    plt.close()
                else:
                    plt.show()
        
        def get_burst_start(self, signals=30, reverse=False):
                #find start of burst using CUSUM method https://doi.org/10.1051/0004-6361:20042620
                #K. Huttunen-Heikinmaa, E. Valtonen and T. Laitinen 2005
                #def cusum(subband,signals=30, reverse=False):
                #find CUSUM for particular subband
                bf_sbs = np.where(np.isclose(freq, self.MHz, atol=1e-2)==True)[0]
                subband = np.sum(I_data[:,bf_sbs[0]:bf_sbs[-1]], axis=1)
                S = np.zeros(subband.shape[0])
                bg_datetime = astropy.time.Time(get_obs_start(bf_file)).datetime
                bg_datetime += timedelta(seconds=t_arr[0])
                dt = t_arr[1]-t_arr[0]
                if not reverse:
                #found by eye
                        bg_datetime_end = datetime(2015,10,17,13,21,40,0)
                        secs = (bg_datetime_end - bg_datetime).total_seconds() 
                        bg_end = int(secs/dt)
                        pre_bg = subband[:bg_end]
                
                else:
                        #foudn by eye
                        bg_datetime_end = datetime(2015,10,17,13,22,20,0)
                        secs = (bg_datetime_end - bg_datetime).total_seconds() 
                        bg_end = int(secs/dt)
                        pre_bg = subband[bg_end:]
                
                mu_a = np.median(pre_bg)#background_define(pre_bg)
                sig_a = np.std(pre_bg)
                mu_d = mu_a +2*sig_a
                k = (mu_d - mu_a)/(np.log(mu_d) - np.log(mu_a))
                if k < 1:
                        h = 1e-1
                else:
                        h = 2e-1
                j = np.zeros(len(subband))
                if not reverse:
                        for i in range(1,subband.shape[0]):
                                S[i] = max(0, subband[i]-k+S[i-1])

                                if S[i] > h:
                                        j[i]=1
                                        if i >signals:
                                                if np.sum(j[i-signals:i])==signals:
                                                        return i-signals
                                if i == subband.shape[0]-1:
                                        return 0
                else:
                        for i in reversed(range(0,subband.shape[0]-1)):
                                S[i] = max(0, subband[i]-k+S[i+1])

                                if S[i] > h:
                                        j[i]=1
                                        if i >signals:
                                                if np.sum(j[i:i+signals])==signals:
                                                        return i+signals


"""
-----------------------------
-----------------------------
Some "helpful" functions
-----------------------------
-----------------------------
"""

def rotate_arr(arr, theta, big_row, big_col):
    rot_mat = np.array(([np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]))
    centre_row, centre_col = big_row/2, big_col/2

    big_arr = np.zeros((big_row+1, big_col+1))
    nrows = arr.shape[0]
    ncols = arr.shape[1]

    for row in range(nrows):
        for col in range(ncols):
            row0, col0 = row - nrows/2, col - ncols/2
            row1, col1 = np.floor(np.matmul(rot_mat, np.array([row0, col0]))).astype(int)
            row1, col1 = int(np.floor(centre_row)) + row1, int(np.floor(centre_col)) + col1

            #print(row, col," : ", row1, col1)
            big_arr[row1, col1] = arr[row, col]


    return big_arr

def rotate_coord(x,y, theta):
    rot_mat = np.array(([np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]))
    return np.floor(np.matmul(rot_mat, np.array([x, y]))).astype(int)

def coord2pix(ra, dec):
        dra = ra_scale[1]-ra_scale[0]
        ddec = dec_scale[1]-dec_scale[0]
        x = np.round((1./dra.hour)*(ra.hour-ra_scale[0].hour))
        y = np.round((1./ddec.deg)*(dec.deg-dec_scale[0].deg))
        x = int(x)
        y = int(y)
        return x, y

def pix2coord(x, y):
        ra = ra_scale[x].hour
        dec = dec_scale[y].deg
        return ra, dec

def fit_2d_gauss(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
        (x, y) = xy
        x0 = float(x0)
        y0 = float(y0)
        a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
        b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
        c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
        g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
        return g.ravel()

"""
-----------------------------
-----------------------------
Spectra related funcions.
Copied from an early script 
and I'm too lazy to implement
them properly
-----------------------------
-----------------------------
"""

def get_data(f,burst_start=datetime(2015,10,17,13,21,16,8), burst_end = datetime(2015,10,17,13,22,31,7) ):
        SAP = f[-22:-19]
        BEAM = f[-17:-14]
        STOKES = f[-12:-11]
        with h5py.File(f, "r") as h5:
                tsamp = h5["/SUB_ARRAY_POINTING_"+ SAP + "/BEAM_"+BEAM +"/COORDINATES/COORDINATE_0"].attrs["INCREMENT"]
                freq =  h5["/SUB_ARRAY_POINTING_"+ SAP + "/BEAM_"+BEAM +"/COORDINATES/COORDINATE_1"].attrs["AXIS_VALUES_WORLD"]*1e-6
                no_samps =      h5["/SUB_ARRAY_POINTING_"+ SAP + "/BEAM_"+BEAM].attrs["NOF_SAMPLES"]

                exptime_start = astropy.time.Time(get_obs_start(f)).datetime
                #approximate guess for burst start
                #burst_start = datetime(2015,10,17,13,21,16,8)
                #burst_end = datetime(2015,10,17,13,22,31,7)
                burst_len = 60
                #get actual index
                burst_start_index = int(np.floor((burst_start-exptime_start).seconds/tsamp))
                burst_end_index = int(np.floor((burst_end-exptime_start).seconds/tsamp))

                dataset = h5["/SUB_ARRAY_POINTING_"+ SAP + "/BEAM_"+BEAM +"/STOKES_" +STOKES]
                data = dataset[burst_start_index:burst_end_index]
                t_arr = tsamp*np.arange(no_samps)
                t_arr = t_arr[burst_start_index:burst_end_index]
        return data, freq, t_arr

def get_obs_start(f):
        SAP = f[-22:-19]
        with h5py.File(f, "r") as h5:
                group = h5["/SUB_ARRAY_POINTING_"+SAP+"/"]
                obs_start = group.attrs["EXPTIME_START_UTC"]
        return obs_start

def plot_bf(f):
        start_time = get_obs_start(f)
        percentile_min = .1
        percentile_max = 99.9
        I_data, freq, t_arr = get_data(f, burst_start=datetime(2015,10,17,13,21,40), burst_end=datetime(2015,10,17,13,22,0))
        dt = t_arr[1]-t_arr[0]
        datetime_start = datetime.strptime(start_time.decode("utf-8")[:-4], "%Y-%m-%dT%H:%M:%S.%f")
        timedelta_zoom_start = timedelta(seconds=t_arr[0])
        datetime_zoom_start = datetime_start + timedelta_zoom_start
        datetime_zoom_end = datetime_start + timedelta(seconds=t_arr[-1])
        xlims = list([datetime_zoom_start, datetime_zoom_end])
        xlims = dates.date2num(xlims)
        fig, ax = plt.subplots(figsize=(7,5))
        bg_sub = I_data/np.mean(I_data, axis=0)
        im = ax.imshow(bg_sub.T, aspect="auto", extent=[xlims[0], xlims[-1], freq[-1], freq[0]],
                                vmax=np.percentile(bg_sub.T, percentile_max), vmin=np.percentile(bg_sub.T, percentile_min))
        ax.xaxis_date()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power", rotation=90)
        date_format = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(date_format)

        plt.title("Sun on "+ start_time.decode("utf-8")[:-20])
        plt.xlabel("Time (UTC) ")
        plt.ylabel("Frequency (MHz)")
        plt.tight_layout()
        plt.savefig(f[:-3]+"_d_spec_burst_zoom1.png")
#       plt.show()
        return im, ax

def stria_fit(t, A, t_0, t_on, t_off):
    b_fit = np.zeros(t.shape)
   # pdb.set_trace()
    t_r = t_0 - t_on
    t_d = t_off - t_0
    for i in range(t.shape[0]):
        if t[i] <= t_0:
            tau = t_r/np.log(2)
        else:
            tau = t_d/np.log(2)

        b_fit[i] = A*np.exp(-(np.abs(t[i]-t_0)/tau))

    return b_fit

def bf_FHWM(f,sb=1800, burst_start=datetime(2015,10,17,13,21,40), burst_end=datetime(2015,10,17,13,22,0)):
    bf, freq, tarr = get_data(f, burst_start, burst_end)
    HM = 0.5*np.max(b[:,sb])
    peak_lhs = np.where((bf[:,sb]-HM)>0)[0]
    peak_rhs = np.where((bf[:,sb]-HM)>0)[-1]
    FWHM = tarr[peak_rhs] - tarr[peak_lhs]
    return FWHM

def freq_to_density(freq):
    omega = (2*np.pi*freq)*(u.rad/u.s)
    n = (omega**2)*((m_e*eps0)/(e.si**2))
    return n.value * 1e6

def Newkirk(freq):
    n = freq_to_density(freq)
    n_0 = 4.2e4
    R = 4.32*(1/np.log10(n/n_0))
#def FWHM_to_l_scale(FWHM,sb):
    return R

def l_scale(freq):
    R = Newkirk(freq)
    return 1/(((4.32*R_sun)/((R*R_sun)**2))*np.log(10))

def oom_source_size(del_freq, freq):
    L = l_scale(freq)
    del_r = 2*L*(del_freq/freq)
    return del_r

"""
------------------------------
------------------------------
Handy functions for running
stuff in parallel (because I
don't know how to do it 
properly)
------------------------------
------------------------------
"""


def pool_class_plot(t):
        return plot_burst(t).plot_just_sun_pix()

def pool_save_gauss_params(t):
#        t_arr = get_data(bf_file)[2]
#        bf_start = get_obs_start(bf_file)
#        bf_start = astropy.time.Time(bf_start).datetime
#        cusum_start = plot_burst(t).get_burst_start()
#        cusum_end = plot_burst(t).get_burst_start(reverse=True)
#        burst_start = bf_start + timedelta(seconds = t_arr[cusum_start])
#        burst_end = bf_start + timedelta(seconds = t_arr[cusum_end]) 
        obs = astropy.time.Time(plot_burst(t).obs_time).datetime
        if (obs >= burst_start) and (obs <= burst_end):
            print("Hello:", t)    
            return plot_burst(t).gauss_params()[0]
        
"""
------------------------------
------------------------------
Plot everything in parallel
then make a movie
------------------------------
------------------------------
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-indir', dest='indir', help='input directory', default='./')
    parser.add_argument('-bf', dest='bf', help='beamformed file', default='/net/romulus.ap.dias.ie/romulus/murphp30_data/typeIII_int/L401005_SAP000_B000_S0_P000_bf.h5')
    parser.add_argument('-start', type=int, dest='start', help='Which file to start at (def=0)', default=0)
    parser.add_argument('-end', type=int, dest='end', help='Whcih file to end at (def=-1)', default=-1)
    #parser.add_argument('-xpix', type=int, dest='xpix', help='How many pixels in x (def=1024)', default=1024)
    #parser.add_argument('-ypix', type=int, dest='ypix', help='How many pixels in y (def=1024)', default=1024)
    args = parser.parse_args()
    indir = args.indir
    f_list = glob.glob(indir+"*.fits")
    f_list.sort()
    #bf_file = args.bf
    #
    #no_pix_x, no_pix_y = args.xpix, args.ypix
    start_f=args.start
    end_f = args.end
    ##pdb.set_trace()
    ##pdb.set_trace()
    if end_f == -1:
            end_f = len(f_list)
    #I_data, freq, t_arr = get_data(bf_file)
    #
    #bf_start = get_obs_start(bf_file)
    #bf_start = astropy.time.Time(bf_start).datetime
    #cusum_start = plot_burst(0).get_burst_start()
    #cusum_end = plot_burst(0).get_burst_start(reverse=True)
    #burst_start = bf_start + timedelta(seconds = t_arr[cusum_start])
    #burst_end = bf_start + timedelta(seconds = t_arr[cusum_end]) 
    #gauss_params_list = []
    #start_f = 270
    #burst_end_f = 
    with Pool(30) as p:
    #       params = p.map(pool_save_gauss_params, range(start_f, end_f))
           p.map(pool_class_plot, range(start_f, end_f))
    #
    #i = 0
    #try:
    #       while params[i] == None:
    #               i+=1
    #except ValueError:
    #       first_burst = i
    #
    #params = [param for param in params if np.sum(param) != None ]
    #last_burst = first_burst + len(params)
    #
    #
    #amps, x0s, y0s, sig_xs, sig_ys, thetas, offsets = [], [], [], [], [], [], []
    #for j in range(len(params)):
    #               amps.append(params[j][0])
    #               x0s.append(params[j][1])
    #               y0s.append(params[j][2])
    #               sig_xs.append(params[j][3])
    #               sig_ys.append(params[j][4])
    #               thetas.append(params[j][5])
    #               offsets.append(params[j][6])
    #
    #
    ##pdb.set_trace()
    images = glob.glob(indir+'*.png')
    images.sort()
    output = indir+"{:.2f}".format(plot_burst(0).MHz)+'MHz_sun.mp4'
    image_path = images[0]
    frame = cv2.imread(image_path)
    cv2.imshow('video', frame)
    height, width, channels = frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output, fourcc, 20., (width, height))
    
    for image in images:
            image_path = image
            frame = cv2.imread(image_path)
            out.write(frame)
            cv2.imshow('video', frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
    
    out.release()
    cv2.destroyAllWindows()
    
    
    
