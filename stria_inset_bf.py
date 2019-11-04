#!/usr/bin/env python
# from matplotlib import rcParams
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.patches import Ellipse, Rectangle
import  h5py
import astropy.time
from astropy.coordinates import Angle
import astropy.units as u
from datetime import datetime, timedelta
import numpy as np
from astropy.io import fits
import sunpy
import sunpy.map
import sunpy.coordinates.sun as sun
#from radiospectra.spectrogram import Spectrogram
import pdb
from plot_fits import LOFAR_to_sun
import argparse
from plot_bf import get_data

# plt.rcParams.update({"font.size": 12})
#plt.rcParams.update({"font.serif": ["Computer Modern Roman"]})


f = "L401005_SAP000_B000_S0_P000_bf.h5"
data, freq, t_arr = get_data(f, datetime(2015,10,17,13,21,0,0),datetime(2015,10,17,13,23,0,0))
dt = t_arr[1] - t_arr[0]



day_start = datetime(2015,10,17,8,00,00)
bf_dt_arr = day_start + timedelta(seconds=1)*t_arr
bf_dt_arr = dates.date2num(bf_dt_arr)
date_format = dates.DateFormatter("%H:%M:%S")
date_format0 = dates.DateFormatter("%M:%S")
bg_data = np.mean(data[:1000,:], axis=0)
data = data/bg_data

# manually found very lazy, sorry
stria_floc = 1992
stria_freq = freq[stria_floc]
stria_time = datetime(2015, 10, 17, 13, 21, 46, 174873)
stria_tloc = int((stria_time - datetime(2015,10,17,13,21,0,0)).total_seconds()/dt)



fig, ax = plt.subplots(figsize=(9,9))
ax.imshow(data[:,1000:].T, aspect='auto', extent=[bf_dt_arr[0], bf_dt_arr[-1], freq[-1], freq[1000]],
	vmin=np.percentile(data[:,1000:],.1), vmax=np.percentile(data[:,1000:],99.9))
rect = Rectangle((bf_dt_arr[stria_tloc-int(5/dt)],freq[ stria_floc-50]), (bf_dt_arr[int(10/dt)]-bf_dt_arr[0]), freq[100]-freq[0], fill=False, color='white' )
ax.add_patch(rect)
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)
# plt.axhline(y=freq[stria_floc], color='r')
# plt.axvline(x=bf_dt_arr[stria_tloc], color='r')
plt.xlabel("Time")
plt.ylabel("Frequency (MHz)")
plt.tight_layout()
# fig.autofmt_xdate()

ax0 = fig.add_axes([0.7,0.17,0.25,0.25])#([0.64,0.2,.25,.25])

ax0.imshow(data[stria_tloc-int(5/dt):stria_tloc+int(5/dt), stria_floc-50:stria_floc+50 ].T, aspect='auto',
	extent=[bf_dt_arr[stria_tloc-int(5/dt)], bf_dt_arr[stria_tloc+int(5/dt)], freq[ stria_floc+50], freq[ stria_floc-50]])
ax0.scatter(bf_dt_arr[stria_tloc],freq[stria_floc],color='r')
ax0.xaxis_date()
ax0.set_xlabel('Time')
ax0.set_ylabel('Frequecny (MHz)')
ax0.xaxis.set_major_formatter(date_format0)
ax0.spines['top'].set_color('white')
ax0.spines['bottom'].set_color('white')
ax0.spines['left'].set_color('white')
ax0.spines['right'].set_color('white')
ax0.xaxis.label.set_color('white')
ax0.yaxis.label.set_color('white')
ax0.tick_params(axis='x', colors='white')
ax0.tick_params(axis='y', colors='white')
for tick in ax0.get_xticklabels():
    tick.set_rotation(45)

plt.savefig("Burst_inset.png")
plt.show()





