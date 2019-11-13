#!/usr/bin/env python
#script to find striae in dynamic spectrum and run wsclean for appropriate SBs
#Author: Pearse Murphy

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
from scipy.signal import savgol_filter, find_peaks
from multiprocessing import Pool
import argparse
import itertools
import subprocess
from plot_fits import plot_one

def get_burst_start(I_data, t_arr, sb,signals=30, reverse=False):
#find start of burst using CUSUM method https://doi.org/10.1051/0004-6361:20042620
#K. Huttunen-Heikinmaa, E. Valtonen and T. Laitinen 2005
#def cusum(subband,signals=30, reverse=False):
#find CUSUM for particular subband
    #print(sb) 
    #bf_sbs =  [1000,1010]#np.where(np.isclose(freq, self.MHz, atol=1e-2)==True)[0]
    subband = I_data[:,sb]#savgol_filter(I_data[:,sb], 11, 3)#np.sum(I_data[:,bf_sbs[0]:bf_sbs[-1]], axis=1)
    bg_datetime = astropy.time.Time(get_obs_start(bf_file)).datetime
    bg_datetime += timedelta(seconds=t_arr[0])
    dt = t_arr[1]-t_arr[0]
    if not reverse:
        bg_datetime_end = datetime(2015,10,17,13,21,40,0)
        if freq[sb] < 35:
            bg_datetime_end = datetime(2015,10,17,13,21,44,0)
        #    m = (26.56-52.24)/(dates.date2num(datetime(2015,10,17,13,21,50)) - dates.date2num(datetime(2015,10,17,13,21,43))) # rough calculation
        #    t_b = freq[sb]/m
        #    bg_datetime_end =  dates.num2date(dates.date2num(bg_datetime_end) + abs(t_b)).replace(tzinfo=None)
        secs = (bg_datetime_end - bg_datetime).total_seconds()# - 2 #minus 2 seconds to account for rough calculation 
        bg_end = int(secs/dt)
        pre_bg = subband[:bg_end]
        S = np.zeros(subband[bg_end:].shape[0])
    else:
        bg_datetime_end = datetime(2015,10,17,13,22,20,0)
        secs = (bg_datetime_end - bg_datetime).total_seconds() 
        bg_end = int(secs/dt)
        pre_bg = subband[bg_end:]
        S = np.zeros(subband[:bg_end].shape[0])
        
    mu_a = np.mean(pre_bg)#background_define(pre_bg)
    sig_a = np.std(pre_bg)
    mu_d = mu_a +3*sig_a
    k = (mu_d - mu_a)/(np.log(mu_d) - np.log(mu_a))
    if k < 1*np.mean(pre_bg):
        h = 0.9*np.mean(pre_bg)#e-1
    else:
        h = 0.8*np.mean(pre_bg)#e-1
    j = np.zeros(len(subband))
    if not reverse:
        for i in range(1,subband[bg_end:].shape[0]):
            S[i] = max(0, subband[bg_end+i]-k+S[i-1])
            if S[i] > h:
                j[i]=1
                if i >signals:
                    if np.sum(j[i-signals:i])==signals:
                        return bg_end+i-signals
            if i == subband[bg_end:].shape[0]-1:
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
                
                #get actual index
                burst_start_index = int(np.floor((burst_start-exptime_start).total_seconds()/tsamp))
                burst_end_index = int(np.ceil((burst_end-exptime_start).total_seconds()/tsamp))

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

def plot_bf(f,oplotx=0, oploty=0,out="default"):
        start_time = get_obs_start(f)
        percentile_min = .1
        percentile_max = 99.9
        b_start=datetime(2015,10,17,13,21,35)
        b_end=datetime(2015,10,17,13,22,0)
        I_data, freq, t_arr = get_data(f, burst_start=b_start,
                                        burst_end=b_end)
        I_data = I_data[:,1000:]
        freq = freq[1000:]
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
        
        if type(oplotx) != int:
            ax.scatter(oplotx, oploty, color='r')

        ax.xaxis_date()
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Intensity (background subtracted)", rotation=90)
        date_format = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(date_format)

        plt.title("Sun on "+ start_time.decode("utf-8")[:-20])
        plt.xlabel("Time (UTC) ")
        plt.ylabel("Frequency (MHz)")
        plt.tight_layout()
        if out =="default":
            plt.savefig(f[:-3]+"_d_spec_burst_zoom.png")
        else:
            plt.savefig(out)
#       plt.show()

def plot_bf_onset(f):
        start_time = get_obs_start(f)
        percentile_min = .1
        percentile_max = 99.9

        b_start=datetime(2015,10,17,13,21,35)
        b_end=datetime(2015,10,17,13,22,0)
        I_data, freq, t_arr = get_data(f, burst_start=b_start,
                                        burst_end=b_end)
        I_data = I_data[:,1000:4698]
        freq = freq[1000:4698]
        dt = t_arr[1]-t_arr[0]
        datetime_start = datetime.strptime(start_time.decode("utf-8")[:-4], "%Y-%m-%dT%H:%M:%S.%f")
        timedelta_zoom_start = timedelta(seconds=t_arr[0])
        datetime_zoom_start = datetime_start + timedelta_zoom_start
        datetime_zoom_end = datetime_start + timedelta(seconds=t_arr[-1])
        xlims = list([datetime_zoom_start, datetime_zoom_end])
        xlims = dates.date2num(xlims)
        
        onset = np.zeros(I_data.shape[1])
        #for i in range(I_data.shape[1]):
        #    onset[i] = t_arr[get_burst_start(i)]
        
        with Pool() as p:
            #onset = p.map(get_burst_start, range(I_data.shape[1]))
            onset = p.starmap(get_burst_start, zip(itertools.cycle([I_data]),itertools.cycle([t_arr]),range(I_data.shape[1]),itertools.cycle([30])))
        
        #pdb.set_trace()
        b_max = np.zeros(len(onset))
#        onset_mask = np.ma.masked_equal(onset,0)
        #for i in range(len(onset)):
            #onset[i] =np.where(I_data[:,i]==np.max(I_data[onset[i]:onset[i]+190,i]))[0][0]
            #onset[i]+200 is about 2 seconds after burst onset 
            #onset[i] += 170
        #pdb.set_trace()
        #onset = savgol_filter(onset, 751,3)
        #onset = onset.astype(int)
        onset = t_arr[onset]
        onset = onset*timedelta(seconds=1)
        onset = onset + datetime_start
        onset = dates.date2num(onset)
 #       onset = np.ma.masked_array(onset, mask=onset_mask.mask)
        
        fig, ax = plt.subplots(figsize=(7,5))
        bg_sub = I_data/np.mean(I_data[:475], axis=0)
        im = ax.imshow(bg_sub.T, aspect="auto", extent=[xlims[0], xlims[-1], freq[-1], freq[0]],
                                vmax=np.percentile(bg_sub.T, percentile_max), vmin=np.percentile(bg_sub.T, percentile_min))
        ax.scatter(onset, freq,color='r',marker='+')
        ax.xaxis_date()
        
        #bg_datetime_end = datetime(2015,10,17,13,21,40,0)
        #bg_datetime_end = dates.date2num(bg_datetime_end)
        #ax.axvline(x=bg_datetime_end)
       

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("Power", rotation=90)
        date_format = dates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(date_format)

        plt.title("Sun on "+ start_time.decode("utf-8")[:-20])
        plt.xlabel("Time (UTC) ")
        plt.ylabel("Frequency (MHz)")
        plt.tight_layout()
        plt.savefig(f[:-3]+"_d_spec_burst_onset5.png")

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

def gauss_1D(x,A,mu,sig):
    return A*np.exp(-((x-mu)**2)/(2*(sig**2)))

def gauss_2D(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
        (x, y) = xy
        x0 = float(x0)
        y0 = float(y0)
        a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
        b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
        c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
        g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
        return g.ravel()


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

def onclick(event):
    global ix, iy
    ix, iy = event.xdata, event.ydata
    print('x = %d, y = %d'%(
        ix, iy))

    global coords
    coords.append((ix, iy))

    if len(coords) == 1:
        fig.canvas.mpl_disconnect(cid)

    return coords

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
    bf_file = args.bf
    
    start_f=args.start
    end_f = args.end
    #plot_bf_onset(bf_file)
    b_start=datetime(2015,10,17,13,21,30)
    b_end=datetime(2015,10,17,13,22,0)
    I_data, freq, t_arr = get_data(bf_file, burst_start=b_start,burst_end=b_end)
    #pdb.set_trace()
    #pdb.set_trace()
    #I_data, freq, t_arr = get_data(bf_file)
    I_data = I_data[:,1000:]#4698] #noise above 1000 and no burst below 4698
    freq = freq[1000:]#4698]
    # bg_sub = I_data/np.mean(I_data[:475], axis=0)
    I_data = I_data/np.mean(I_data[:475], axis=0)
#    plt.plot(I_data[:,2111])
#    plt.axvline(x=get_burst_start(I_data,t_arr,2111), color='r')
#    plt.axvline(x=get_burst_start(I_data, t_arr,2111)+200, color='r')
    bf_start = get_obs_start(bf_file)
    bf_start = astropy.time.Time(bf_start).datetime
    t_arrdt = t_arr*timedelta(seconds=1)
    times = bf_start + t_arrdt
    #cusum_start = get_burst_start()
    ##cusum_end = get_burst_start(reverse=True)
    ##burst_start = bf_start + timedelta(seconds = t_arr[cusum_start])
    ##burst_end = bf_start + timedelta(seconds = t_arr[cusum_end])
    times = dates.date2num(times)
     
        
#    with Pool() as p:
#        #onset = p.map(get_burst_start, range(I_data.shape[1]))
#        onset = p.starmap(get_burst_start, zip(itertools.cycle([I_data]),itertools.cycle([t_arr]),range(I_data.shape[1]),itertools.cycle([1000])))
#
#    plot_bf_onset(bf_file)
    """    
    with Pool() as p:
        onset = p.starmap(get_burst_start, zip(itertools.cycle([I_data]),itertools.cycle([t_arr]),range(I_data.shape[1]),itertools.cycle([30])))

#
#    #onset_mask = np.ma.masked_equal(onset,0)
    for i in range(len(onset)):
        onset[i] =np.where(I_data[:,i]==np.max(I_data[onset[i]:onset[i]+190,i]))[0][0]
    onset = savgol_filter(onset, 751,3)
    onset = onset.astype(int)
    b_spine = np.zeros(len(onset))
    #b_rib = np.zeros(len(onset)) #because it's not the spine of the burst, it must be the rib
    for i in range(len(onset)):
        #b_rib[i] = bg_sub[onset[i],i]
        b_spine[i] = bg_sub[onset[i],i]#bg_sub[np.where(bg_sub[:,i]==np.max(bg_sub[onset[i]:onset[i]+250,i]))[0][0],i]
#    
    """   
#    """------
#    Warning, next 2 lines involve me saying
#    "That looks about right". Worth investigating further
#    ------"""
#    
#   smooth_spine = savgol_filter(b_spine, 41, 4) #seemed to work best
#   striae, props = find_peaks(smooth_spine[:1500], distance=7, width=[5,60]) #seemed to work best
#
#
#
#    plt.figure()
#    plt.plot(freq[:1500],b_spine[:1500], label="Intensity")
#    plt.plot(freq[:1500],smooth_spine[:1500], label="Intensity smoothed")
#    plt.plot(freq[striae], smooth_spine[striae], 'ro', label="Peak Position")
#    plt.hlines(props["width_heights"], freq[props["left_ips"].astype(int)], freq[props["right_ips"].astype(int)], color='r', label="Peak Width") 
#   #int(np.round()) instead of .astype(int) maybe? Only for plot so I don't really care
##    plt.figure()
##    plt.plot(freq, bg_sub[onset[3000],:])
##    plt.axvline(x=freq[3000], color='r')
#    plt.xlabel("Frequency (MHz)")
#    plt.ylabel("Intensity (bg subtracted)")
#    plt.title("Intensity along burst")
#    plt.legend()
##    
##    
##    
#    plt.savefig("Striae_fit_zoom.png")    
    """    
    fig,ax = plt.subplots(figsize=(16,9))
    percentile_max = 99.
    percentile_min = 1.
    t_arr_start = 1000
    freq_end = 1800
    ax.imshow(bg_sub[t_arr_start:,:freq_end].T, aspect="auto",vmax=np.percentile(bg_sub.T, percentile_max), vmin=np.percentile(bg_sub.T, percentile_min))#,extent=[t_arr[1000],t_arr[-1], freq[1800], freq[0]])
    
    
    coords = []
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.show()
    
    click_freq = np.zeros(len(coords),dtype=int)
    click_tarr = np.zeros(len(coords),dtype=int)
    #array locations of clicks
    """
    #click_tarr = np.array([1931, 1973, 1821, 1858, 1648, 1551, 1654, 1619, 1619])#, 1615])
    #click_freq = np.array([19, 99, 319, 486, 561, 750, 832, 958, 1067])#, 1280])
    
    click_tarr = np.array([2336, 2261, 2323, 2200, 2287, 2225, 2159, 2156, 2155, 2159])
    click_freq = np.array([965,991, 1025, 1222,1300,1472,1664, 1708, 1773, 1796])
    
    #click_tarr = np.array([2200])
    #click_freq = np.array([1222])
    #for i in range(len(coords)):
    #    click_tarr[i] = t_arr_start + int(np.round(coords[i][0]))
    #    click_freq[i] = int(np.round(coords[i][1]))
    np.savez("striae/clicks/click_locations.npz",click_tarr, click_freq) 

    #actual locations of clicks
    striae_tarr = t_arr[click_tarr]
    striae_freq = freq[click_freq]
   
    striae_dt = bf_start + striae_tarr*timedelta(seconds=1)
    
    
    
    #plt.savefig("d_spec_click.png")
   
    #chosen manually
    #find array location of clicked data points
    
    """
    Find F and Delta F from dynamic spectrum
    load txt file with interferometric freqs
    compare

    Everything after this should probably be in functions
    but I was only figuring out what I was doing as I was
    writing....

    """
    
    d_f = freq[1]-freq[0] 
    #del_f = d_f*props["widths"]
    #f = freq[striae]
    f = striae_freq
    bg_len = 15 #extra data around peak
    sig_bound = 0.2 #bound on guassian sigma for fit
    #pdb.set_trace()
    #bg_sub = savgol_filter(bg_sub, 15, 4)
    mus = np.zeros(len(click_freq))
    sigs = np.zeros(len(click_freq))
#     for ck in range(len(click_freq)):
#         if (click_freq[ck] - bg_len) > 0:
#             #click_max = np.max(bg_sub[click_tarr[ck],click_freq[ck]-bg_len:click_freq[ck]+bg_len]) #max value near click
#             #max_loc = np.where(bg_sub[click_tarr[ck],click_freq[ck]-bg_len:click_freq[ck]+bg_len] == click_max)
#             #max_loc = click_freq[ck] - bg_len + max_loc
#             #click_freq[ck] = max_loc
#             #striae_freq[ck] = freq[max_loc]
#             popt,pcov = opt.curve_fit(gauss_1D, freq[click_freq[ck]-bg_len:click_freq[ck]+bg_len], 
#                 bg_sub[click_tarr[ck],click_freq[ck]-bg_len:click_freq[ck]+bg_len],
#                 p0=[bg_sub[click_tarr[ck],click_freq[ck]], striae_freq[ck],0.01],
#                 bounds=([0,striae_freq[ck]-.1,0],[np.inf,striae_freq[ck]+.1,sig_bound]))
#             gauss1 = gauss_1D(freq[click_freq[ck]-bg_len:click_freq[ck]+bg_len], popt[0], popt[1], popt[2])
#             plt.figure()
#             plt.plot(freq[click_freq[ck]-bg_len:click_freq[ck]+bg_len], bg_sub[click_tarr[ck],click_freq[ck]-bg_len:click_freq[ck]+bg_len]) 
#             plt.plot(striae_freq[ck], bg_sub[click_tarr[ck],click_freq[ck]], 'ro')
#             plt.plot(freq[click_freq[ck]-bg_len:click_freq[ck]+bg_len],gauss1)  
           
#             plt.title("Striae at {} MHz".format(striae_freq[ck]))
#             plt.xlabel("Frequency (MHz)")
#             plt.ylabel("Power (bg subtracted)")
#             plt.savefig("striae/Striae_{}_MHz.png".format(striae_freq[ck]))
#         elif (click_freq[ck] - bg_len) < 0:
#             #click_max = np.max(bg_sub[click_tarr[ck],:click_freq[ck]+bg_len]) #max value near click
#             #max_loc = np.where(bg_sub[click_tarr[ck],:click_freq[ck]+bg_len] == click_max)[0]
# #            max_loc = bg_len + max_loc
#             #click_freq[ck] = max_loc
#             #striae_freq[ck] = freq[max_loc]
#             popt,pcov = opt.curve_fit(gauss_1D, freq[:click_freq[ck]+bg_len], 
#                 bg_sub[click_tarr[ck],:click_freq[ck]+bg_len],
#                 p0=[bg_sub[click_tarr[ck],click_freq[ck]], striae_freq[ck],0.01],
#                 bounds=([0,striae_freq[ck]-.1,0],[np.inf,striae_freq[ck]+.1,sig_bound]))
            
#  #           pdb.set_trace() 
#             gauss1 = gauss_1D(freq[:click_freq[ck]+bg_len], popt[0], popt[1], popt[2])
#             plt.figure()
#             plt.plot(freq[:click_freq[ck]+bg_len], bg_sub[clck_tarr[ck],:click_freq[ck]+bg_len]) 
#             plt.plot(striae_freq[ck], bg_sub[click_tarr[ck],click_freq[ck]], 'ro')
#             plt.plot(freq[:click_freq[ck]+bg_len],gauss1)  
#             plt.title("Striae at {} MHz".format(striae_freq[ck]))
#             plt.xlabel("Frequency (MHz)")
#             plt.ylabel("Power (bg subtracted)")
#             plt.savefig("striae/Striae_{}_MHz.png".format(striae_freq[ck]))
#         elif (click_freq[ck] + bg_len) > len(f):
#             #click_max = np.max(bg_sub[click_tarr[ck],click_freq[ck]-bg_len:]) #max value near click
#             #max_loc = np.where(bg_sub[click_tarr[ck],click_freq[ck]-bg_len:] == click_max)
#             #max_loc = click_freq[ck] - bg_len + max_loc
#             #click_freq[ck] = max_loc
#             #striae_freq[ck] = freq[max_loc]
#             popt,pcov = opt.curve_fit(gauss_1D, freq[click_freq[ck]-bg_len:], 
#                 bg_sub[click_tarr[ck],click_freq[ck]-bg_len:],
#                 p0=[bg_sub[click_tarr[ck],click_freq[ck]], striae_freq[ck],0.01],
#                 bounds=([0,striae_freq[ck]-.1,0],[np.inf,striae_freq[ck]+.1,sig_bound]))
        
#             gauss1 = gauss_1D(freq[click_freq[ck]-bg_len:], popt[0], popt[1], popt[2])
#             plt.figure()
#             plt.plot(freq[click_freq[ck]-bg_len:], bg_sub[click_tarr[ck],click_freq[ck]-bg_len:]) 
#             plt.plot(striae_freq[ck], bg_sub[click_tarr[ck],click_freq[ck]], 'ro')
#             plt.plot(freq[click_freq[ck]-bg_len:],gauss1)  
#             plt.title("Striae at {} MHz".format(striae_freq[ck]))
#             plt.xlabel("Frequency (MHz)")
#             plt.ylabel("Power (bg subtracted)")
#             plt.savefig("striae/Striae_{}_MHz.png".format(striae_freq[ck]))
#         mus[ck] = popt[1]
#         sigs[ck] = popt[2]
# #    plt.close('all')
#     np.save('striae_centres',mus)
#     np.save('striae_widths', sigs)
    
    #pdb.set_trace()
    with open("SB_to_freqs.txt", "r") as sbs:
        sbs_str = sbs.read()

    sbs_list = sbs_str.split("msoverview: Version 20110407GvD")
    del sbs_list[0] #it's empty anyway

    int_freqs = np.zeros(len(sbs_list))
    for i in range(len(sbs_list)):
        freq_s = sbs_list[i].find("TOPO") + 4
        freq_e = sbs_list[i].find("195.312")
        int_freqs[i] = float(sbs_list[i][freq_s:freq_e])
    np.save("int_freqs", int_freqs)
#    fig, ax = plt.subplots()
#    im = ax.imshow(I_data.T, aspect='auto', vmin=np.percentile(I_data, 1), vmax=np.percentile(I_data, 99), extent=[times[0],times[-1], freq[-1], freq[0]])
#    for i_f in int_freqs:
#        if i_f < freq[-1]:
#           ax.scatter(times[1000],i_f, color='r')
#            # ax.axhline(y=i_f, color='r')

    fig, ax = plt.subplots()
    #a couple of magic numbers to zoom in on clicked region
    im = ax.imshow(I_data[2000:2500,900:1900].T, aspect='auto', vmin=np.percentile(I_data[2000:2500,900:1900], 1), vmax=np.percentile(I_data[2000:2500,900:1900], 99), extent=[times[2000],times[2500], freq[1900], freq[900]])
    ax.scatter(dates.date2num(striae_dt), striae_freq, color='r')
    ax.xaxis_date()
    
    for i_f in int_freqs:
        if i_f > freq[900] and i_f < freq[1900]:
    #       ax.scatter(times[2250],i_f, color='r',marker='.')
           ax.axhline(y=i_f, color='r')
    #bg_datetime_end = datetime(2015,10,17,13,21,40,0)
    #bg_datetime_end = dates.date2num(bg_datetime_end)
    #ax.axvline(x=bg_datetime_end)
    
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Intensity (au)", rotation=90)
    date_format = dates.DateFormatter("%H:%M:%S")
    #ax.xaxis.set_major_formatter(date_format)
    
    plt.title("Zoom in of Striated Burst")
    plt.xlabel("Time (UTC) ")
    plt.ylabel("Frequency (MHz)")
    plt.tight_layout()
    #plt.savefig("burst2_zoom.png")
    #plot_bf(bf_file)
    plt.show()
    striae_SB = np.zeros(len(click_freq))#np.zeros(len(f))
    """
    for i in range(len(striae_SB)):
        int_striae = np.where(np.isclose(f[i],int_freqs, rtol=4e-3)==True)[0]#int_freqs[np.isclose(f[i],int_freqs, rtol=.5e-2)] 
        if len(int_striae) > 1:
            diff = f[i] - int_freqs[int_striae]
            striae_SB[i] = int_striae[np.where(diff == np.min(diff))][0]
        else:
            striae_SB[i] = int_striae[0]clic
    """

    for i in range(len(striae_SB)):
        int_striae = np.where(np.isclose(striae_freq[i],int_freqs, rtol=4e-3)==True)[0]#int_freqs[np.isclose(f[i],int_freqs, rtol=.5e-2)] 
        if len(int_striae) > 1:
            diff = striae_freq[i] - int_freqs[int_striae]
            striae_SB[i] = int_striae[np.where(diff == np.min(diff))][0]
        else:
            striae_SB[i] = int_striae[0]

    with open("striae_SB.txt", "w") as sbs:
        for stria in striae_SB.astype(int):
            sbs.write(str(stria).zfill(3))
            sbs.write('\n')
    #pdb.set_trace()    
    #flag data based on SBs selected. Baselines to be flagged determined manually 
    #subprocess.run("/mnt/murphp30_data/typeIII_int/run_DPPP_flag.sh")

    """
    find scale, width and interval parameters for WSClean
    should really make it it's own function
    """
    #time sampling of interferometry 
    ntimes = 86280
    tot_elapsed_time = 21600
    int_start = datetime(2015,10,17,8,00,00)
    int_dt = tot_elapsed_time/ntimes
    #b_max_timedelta = t_arr[onset]*timedelta(seconds=1)
    #b_max_datetime = int_start+b_max_timedelta
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
    mgain_val = "0.8" 
    size = "-size"
    scale  = "-scale"
    pol = "-pol"
    pol_val = "I" 
    data_column = "-data-column"
    data_column_val = "CORRECTED_DATA"
    intervals_out = "-intervals-out"
    intervals_out_val = "80"#"20" 
    interval = "-interval"
    use_diff_lofar_beam = "-use-differential-lofar-beam"
    multiscale = "-multiscale"
    fit_beam = "-fit-beam"
    make_psf = "-make-psf"
    beamfitsize = "-beam-fitting-size"
    beamfitsize_val ="4"
    niter = "-niter"
    niter_val = "0"
    auto_threshold = "-auto-threshold"
    auto_threshold_val = "3"
    circular_beam = "-circularbeam"
    multiscale = "-multiscale"
    name = "-name"
    
    i_sb = 0
    striae_SB = np.arange(239,244)
    for SB in striae_SB:
        l = c.value/(int_freqs[int(SB)]*1e6)
        B = 84974.55079 #gotten manually from msoverview verbose
        theta_deg = Angle((l/B)*u.rad)
        theta_asec = theta_deg.arcsec
        scale_val = round(theta_asec/4,4)
        im_ang_width = Angle(10000*u.arcsec) #make all images with an angular width of 10000 arcsecs
        im_pix_width = im_ang_width/(theta_deg.to(u.arcsec)/4)
        #interval_start = int(t_arr[onset[int(SB)]]/int_dt)
        #pdb.set_trace()
        """
        Intervals found by eye :/
        """
        interval_start = 77092#int(np.round(striae_tarr[i_sb]/int_dt))-20
        interval_end = 77092+80#int(np.round(striae_tarr[i_sb]/int_dt)) + 20 #interval_start + 10 #only image one time sample
        
        #adjust values to run with subprocess

        
        interval_start = str(interval_start)
        interval_end = str(interval_end)
        size_x = str(int(im_pix_width.value))
        if len(str(scale_val)) < 6:
            scale_val = str(scale_val)+"0asec"
        else:
            scale_val = str(scale_val)+"asec"
        name_val = "dirty_images/SB{}/wsclean-SB{}".format(str(int(SB)).zfill(3),str(int(SB)).zfill(3))#"striae/all/burst2/SB{}/wsclean-SB{}".format(str(int(SB)).zfill(3),str(int(SB)).zfill(3))
        ms = "/mnt/murphp30_data/typeIII_int/L401003_SB{}_uv.dppp.MS".format(str(int(SB)).zfill(3))
        #pdb.set_trace()
        
        try:
            print(["wsclean", j, j_val, mem, mem_val, no_reorder,
            no_update_model_required, weight, briggs, briggs_val, mgain,
            mgain_val, size, size_x, size_x, scale, scale_val, pol, pol_val,
            data_column, data_column_val,auto_threshold,auto_threshold_val,#multiscale,
            beamfitsize, beamfitsize_val,# circular_beam,
            interval, interval_start, interval_end,
            intervals_out, intervals_out_val,use_diff_lofar_beam,
            fit_beam, make_psf, niter, niter_val,name,name_val , ms])
            
            subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
            no_update_model_required, weight, briggs, briggs_val, mgain,
            mgain_val, size, size_x, size_x, scale, scale_val, pol, pol_val,
            data_column, data_column_val,auto_threshold,auto_threshold_val,#multiscale,
            beamfitsize, beamfitsize_val,# circular_beam,
            interval, interval_start, interval_end, intervals_out, intervals_out_val,
            use_diff_lofar_beam,
            fit_beam, make_psf, niter, niter_val,name,name_val , ms], check=True)
        except subprocess.CalledProcessError:
            size_x = str(int(im_pix_width.value)+1) #increase size by 1

            subprocess.run(["wsclean", j, j_val, mem, mem_val, no_reorder,
            no_update_model_required, weight, briggs, briggs_val, mgain,
            mgain_val, size, size_x, size_x, scale, scale_val, pol, pol_val,
            data_column, data_column_val,auto_threshold,auto_threshold_val,#multiscale,
            beamfitsize, beamfitsize_val,# circular_beam,
            interval, interval_start, interval_end,  intervals_out, intervals_out_val,use_diff_lofar_beam,
            fit_beam, make_psf, niter, niter_val,name,name_val , ms])
        #for filetype in ["dirty", "image", "psf"]:
        #    plot_one(name_val+"-{}.fits".format(filetype))
        #    plt.savefig(name_val+"-{}.png".format(filetype))
        #    plt.close()
        
        i_sb+=1
        
