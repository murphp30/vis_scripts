#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib import dates
import  h5py
import astropy.time
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
import sunpy
import sunpy.map
from radiospectra.spectrogram import Spectrogram
import pdb
from plot_fits import LOFAR_to_sun

bf_file = "/mnt/murphp30_data/typeIII_int/L401005_SAP000_B000_S0_P000_bf.h5"
in_fits = "striae/clicks/burst2/SB091/wsclean-SB091-t0010-image.fits"



def int_to_bf_freq(int_freq, freq_arr, reltol=1e-3):
	#gives index of freq_arr of closest value to int_freq
	close_freqs = np.where(np.isclose(int_freq,freq_arr,reltol)==True)[0]
	if len(close_freqs) > 1:
		diff = np.abs(int_freq - freq_arr[close_freqs])
		bf_freq = close_freqs[np.where(diff == np.min(diff))][0]
	else:
		bf_freq = close_freqs[0]

	return bf_freq

class stria:
	
	def __init__(self,int_fits):
		self.int_fits = int_fits
		self.bf = bf_file
		smap = sunpy.map.Map(self.int_fits, plot_settings={'cmap':'viridis'})
		smap.meta['wavelnth'] = smap.meta['crval3']/1e6
		smap.meta['waveunit'] = "MHz"
		self.smap = LOFAR_to_sun(smap)

		with fits.open(int_fits) as hdu:
			self.stria_int_freq = hdu[0].header['CRVAL3']
			stria_peak_time = hdu[0].header["DATE-OBS"]
			self.stria_peak_timeobj = astropy.time.Time(stria_peak_time)
	
		with h5py.File(self.bf, "r") as h5:
			tsamp = h5["/SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES/COORDINATE_0"].attrs["INCREMENT"]
			self.dt = astropy.time.TimeDelta(tsamp,format='sec')
			self.freq_arr =  h5["/SUB_ARRAY_POINTING_000/BEAM_000/COORDINATES/COORDINATE_1"].attrs["AXIS_VALUES_WORLD"]
			self.df = self.freq_arr[1]-self.freq_arr[0]
			obs_start = h5["/SUB_ARRAY_POINTING_000"].attrs["EXPTIME_START_UTC"]
			self.obs_start_timeobj = astropy.time.Time(obs_start)

	def get_stria(self):

		stria_click_time, stria_click_freq = self.click_coords()

		stria_peak_index = int(np.round((stria_click_time - self.obs_start_timeobj).sec/self.dt.sec))#int(np.round((self.stria_peak_timeobj - self.obs_start_timeobj).sec/self.dt.sec))
		stria_start_index = stria_peak_index - int(np.round(2./self.dt.sec)) #start 2 seconds before peak
		stria_end_index = stria_peak_index + int(np.round(2./self.dt.sec)) #end 2 seconds before peak
		
		tarr = np.arange(self.obs_start_timeobj+stria_start_index*self.dt, 
			self.obs_start_timeobj+(stria_end_index-1)*self.dt,self.dt)

		stria_bf_freq = np.where(self.freq_arr == stria_click_freq)[0][0]#int_to_bf_freq(self.stria_int_freq,self.freq_arr)
		stria_bf_freq_start = stria_bf_freq - int(np.round(0.5e6/self.df)) #start half MHz below peak
		stria_bf_freq_end = stria_bf_freq + int(np.round(0.5e6/self.df)) #start half MHz above peak
		
		freq = self.freq_arr[stria_bf_freq_start:stria_bf_freq_end]

		with h5py.File(self.bf, "r") as h5:
			dataset = h5["/SUB_ARRAY_POINTING_000/BEAM_000/STOKES_0"]
			data = dataset[stria_start_index:stria_end_index,stria_bf_freq_start:stria_bf_freq_end]
	#pdb.set_trace()
		return data, tarr, freq

	def click_coords(self):
		""" 
		The following are taken from manual clicks run in 
		plot_bf.py. A proper version of this code should 
		load in these values from a .npy save
		Returns value of click location in time and frequency
		"""

		click_tarr = np.array([2336, 2261, 2323, 2200, 2287, 2225, 2159, 2156, 2155, 2159])
		click_freq = np.array([965,991, 1025, 1222,1300,1472,1664, 1708, 1773, 1796])

		click_start_datetime  = astropy.time.Time('2015-10-17T13:21:30.000')
		click_start_offset = 1000
		click_start_freq = 1000
		click_freq = click_freq + click_start_freq
		
		#find where click best matches stria
		index_in_click_freq = int_to_bf_freq(self.stria_int_freq, self.freq_arr[click_freq],1e-2)
		index_in_freq_arr = click_freq[index_in_click_freq] #find in total frequency array

		click_freq_val = self.freq_arr[index_in_freq_arr]

		click_time_val = click_start_datetime + (self.dt*click_tarr[index_in_click_freq])

		return click_time_val, click_freq_val
	def plot(self):
		data, tarr, freq = self.get_stria()
		freq = freq*1e-6 #frequency to MHz
		t_lims = [tarr[0].datetime, tarr[-1].datetime]
		t_lims = dates.date2num(t_lims)

		fig = plt.figure(figsize=(7,7))
		self.smap.plot()
		self.smap.draw_limb()
		ax = fig.add_axes([0.64,0.2,.25,.25])
		im = ax.imshow(data.T,aspect="auto", extent=[t_lims[0], t_lims[-1], freq[-1], freq[0]])
		ax.scatter(dates.date2num(self.click_coords()[0].datetime), 1e-6*self.click_coords()[1], color='r')
		ax.xaxis_date()
		ax.set_xlabel('Time')
		ax.set_ylabel('Frequecny (MHz)')
		
		ax.spines['top'].set_color('white')
		ax.spines['bottom'].set_color('white')
		ax.spines['left'].set_color('white')
		ax.spines['right'].set_color('white')
		ax.xaxis.label.set_color('white')
		ax.yaxis.label.set_color('white')
		ax.tick_params(axis='x', colors='white')
		ax.tick_params(axis='y', colors='white')

		#cbar = fig.colorbar(im, ax=ax)

		date_format = dates.DateFormatter("%M:%S")
		ax.xaxis.set_major_formatter(date_format)
		
		#fig.autofmt_xdate()
		plt.xticks(rotation=30)

if __name__ == "__main__":
	s = stria(in_fits)
	s.plot()
	plt.show()