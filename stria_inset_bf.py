#!/usr/bin/env python
#Used to create Figure 1 in Murphy et al. 2020



# from matplotlib import rcParams
# rcParams['font.family'] = 'serif'
# rcParams['font.serif'] = ['Times New Roman']
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
from matplotlib import dates
from matplotlib.patches import Ellipse, Rectangle, ConnectionPatch
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as path_effects
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
from sunpy.instr.goes import flux_to_flareclass, flareclass_to_flux
#from radiospectra.spectrogram import Spectrogram
import pdb
from plot_fits import LOFAR_to_sun
import argparse
from plot_bf import get_data
from plot_vis import LOFAR_vis

import sunpy.timeseries as ts
from sunpy.net import Fido, attrs as a
from sunpy.time import TimeRange, parse_time

import warnings
warnings.filterwarnings("ignore", "/Users/murphp30/anaconda3/envs/condasun/lib/python3.7/functools.py:840")

files = "go1520151017.fits" #"/Users/murphp30/Documents/Postgrad/my_papers/20151017/GOES/go1520151017.fits" #Fido.fetch(result, path="./")



# plt.rcParams.update({"font.size": 12})
#plt.rcParams.update({"font.serif": ["Computer Modern Roman"]})
parser = argparse.ArgumentParser()
parser.add_argument('--vis_file', dest='vis_file', help='Input visibility file. \
	Must be npz file of the same format as that created by vis_to_npy.py', default='SB076MS_data.npz')
parser.add_argument('--peak', dest='peak', type=int, help='user defined peak', default=28)
args = parser.parse_args()
vis_file = args.vis_file

f = "L401005_SAP000_B000_S0_P000_bf.h5"
data, freq, t_arr = get_data(f, datetime(2015,10,17,13,21,0,0),datetime(2015,10,17,13,23,0,0))
dt = t_arr[1] - t_arr[0]
df = freq[1] - freq[0]


day_start = datetime(2015,10,17,8,0,0)
bf_dt_arr = day_start + timedelta(seconds=1)*t_arr
bf_dt_arr = dates.date2num(bf_dt_arr)
date_format_goes = dates.DateFormatter("%H:%M") #dates.DateFormatter("%M:%S.%f")
date_format_bf = dates.DateFormatter("%H:%M:%S") #dates.DateFormatter("%M:%S.%f")
date_format0 = dates.DateFormatter("%M:%S")
bg_data = np.mean(data[:1000,:], axis=0)
data = data/bg_data

# manually found very lazy, sorry
#vis_file = "SB076MS_data.npz"
peak_dict = {"059":38, "117":50, "118":50, "119":50, "120":50, "125":49, "126":50, "127":50, "130":49, "133":47, "160":23}
try:
	peak = peak_dict[vis_file[2:5]]
except KeyError:
	peak = args.peak
q_t = 1199
stria_vis = LOFAR_vis(vis_file, q_t+peak)
stria_floc = np.where(freq == stria_vis.freq*1e-6 +(df/2))[0][0]
# stria_freq = freq[stria_floc]
stria_time = stria_vis.time#datetime(2015, 10, 17, 13, 21, 46, 174873)
stria_tloc = int((stria_time - datetime(2015,10,17,13,21,0,0)).total_seconds()/dt)

goes_ts = ts.TimeSeries(files, source="XRS")
bf_tr = TimeRange(datetime(2015,10,17,13,21,0,0),datetime(2015,10,17,13,23,0,0))
tr = TimeRange("2015/10/17T12:00:00", "2015/10/17T14:00:00")
goes_tr = goes_ts.truncate(tr)
#date_format = dates.DateFormatter("%H:%M:%S")

fig = plt.figure(figsize=(9,12))
gs = GridSpec(2,1, height_ratios=[1,2])

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])

gax = goes_tr.plot(ax1, legend=False, fontsize=14, rot=0)
#gax.legend([str(0.5)+'-'+str(4.0)+' '+r'$\AA$',str(1.0)+'-'+str(8.0)+' '+r'$\AA$'])
#ax.axvline(parse_time("2015/10/17T13:21:00").plot_date, color="r")
v0 = gax.axvline(bf_tr.start.plot_date, color="r")
v1 = gax.axvline(bf_tr.end.plot_date, color="r")
gax.xaxis.set_major_formatter(date_format_goes)
#gax.set_title("GOES Xray Flux")
gax.set_xlabel("Time (UTC)",fontsize=14)
gax.set_ylabel(r"Watts m$^{-2}$",fontsize=14)
gax.text(0.05,0.9,'a) GOES Xray Flux', fontdict={'size':14}, transform=gax.transAxes)
gax.set_yscale("log")
gax2 = ax1.twinx()
gax2.set_yscale("log")
gax.set_ylim(1e-10, 1e-3)
gax2.set_ylim(1e-10, 1e-3)
gax2.set_yticks((1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3))#((1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2))
gax2.set_yticklabels((' ', ' ', 'A', 'B', 'C', 'M', 'X', ' '))#((' ', 'A', 'B', 'C', 'M', 'X', ' '))
handles, labels = gax.get_legend_handles_labels()
handles.reverse()
labels.reverse()
gax.yaxis.grid(True, 'major')
gax.legend(handles, labels)
# for tick in gax.get_xticklabels():
#     tick.set_rotation(0)
ax2.imshow(data[:,1000:].T, aspect='auto', extent=[bf_dt_arr[0], bf_dt_arr[-1], freq[-1], freq[1000]],
	vmin=np.percentile(data[:,1000:],.1), vmax=np.percentile(data[:,1000:],99.9),cmap='viridis')
rect = Rectangle((bf_dt_arr[stria_tloc-int(5/dt)],freq[ stria_floc-50]), (bf_dt_arr[int(10/dt)]-bf_dt_arr[0]), freq[100]-freq[0], fill=False, color='white' )
ax2.add_patch(rect)
ax2.set_xticklabels(ax2.get_xticklabels(),fontsize=14)

ax2.xaxis_date()
ax2.xaxis.set_major_formatter(date_format_bf)
# plt.axhline(y=freq[stria_floc], color='r')
# plt.axvline(x=bf_dt_arr[stria_tloc], color='r')
ax2.set_xlabel("Time (UTC)",fontsize=14)
ax2.set_ylabel("Frequency (MHz)",fontsize=14)
ax2.set_yticklabels([*ax2.get_yticks()], fontsize=14)
text2 = ax2.text(0.05,0.9,'b) LOFAR Dynamic Spectrum', fontdict={'size':14,'color':'w'}, transform=ax2.transAxes)
text2.set_path_effects([path_effects.Stroke(linewidth=1, foreground='k'),path_effects.Normal()])
#plt.tight_layout()
# fig.autofmt_xdate()

xy0 = v0.get_data()
xy0 = (xy0[0][0],xy0[1][0])
coords0 = gax.get_xaxis_transform()
con_axes0 = ConnectionPatch(xyA=xy0, xyB=(0,1), coordsA=coords0, coordsB="axes fraction", axesA=gax, axesB=ax2, arrowstyle="-", color="r")

xy1 = v1.get_data()
xy1 = (xy1[0][0],xy1[1][0])
coords1 = gax.get_xaxis_transform()
con_axes1 = ConnectionPatch(xyA=xy1, xyB=(1,1), coordsA=coords1, coordsB="axes fraction", axesA=gax, axesB=ax2, arrowstyle="-", color="r")

ax2.add_artist(con_axes0)
ax2.add_artist(con_axes1)




ax0 = fig.add_axes([0.7, 0.12, 0.25, 0.25])#inset_axes(fig, width="25%", height="25%", loc=4)##([0.7,0.17,0.25,0.25])#([0.64,0.2,.25,.25])

rectbbox = rect.get_bbox()
rect_xy0 = np.array((rectbbox.x0,rectbbox.y1)) #data coords
rect_xy1 = np.array((rectbbox.x1,rectbbox.y0)) #data coords

ax0_pos = ax0.get_position()
ax_xy0 = np.array((ax0_pos.x0, ax0_pos.y0))#(0.68,0.11)#ax0_pos.bounds[:2] #axes coords
ax_xy1 = np.array((ax0_pos.x1, ax0_pos.y1))#(0.68,0.11)#ax0_pos.bounds[:2] #axes coords

con_axes2 = ConnectionPatch(
        xyA=rect_xy0, coordsA="data",
        xyB=ax_xy0, coordsB=ax2.transAxes,
        color='w', arrowstyle='-'
        )

con_axes3 = ConnectionPatch(
        xyA=rect_xy1, coordsA="data",
        xyB=ax_xy1, coordsB=ax2.transAxes,
        color='w', arrowstyle='-'
        )

ax2.add_artist(con_axes2)
ax2.add_artist(con_axes3)

ax0.imshow(data[stria_tloc-int(5/dt):stria_tloc+int(5/dt), stria_floc-50:stria_floc+50 ].T, aspect='auto',
	extent=[bf_dt_arr[stria_tloc-int(5/dt)], bf_dt_arr[stria_tloc+int(5/dt)], freq[ stria_floc+50], freq[ stria_floc-50]],cmap='viridis')
ax0.scatter(bf_dt_arr[stria_tloc],freq[stria_floc],color='w', marker='+')
#ax0.axvline(bf_dt_arr[stria_tloc], color='w', ls='--')
#ax0.axhline(freq[stria_floc], color='w', ls='--')

ax0.xaxis_date()
ax0.set_xlabel('Time (UTC)')
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


plt.tight_layout()
#plt.savefig("/Users/murphp30/Documents/Postgrad/my_papers/20151017/images/Burst_inset_goes.png")
plt.savefig("Burst_inset_goes.png")
#plt.show()





