#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import icrs_to_helio as icrs_to_helio
import astropy.units as u
import sunpy
from astropy.io import fits
from astropy import wcs
from sunpy.map import Map
from astropy.coordinates import SkyCoord
from datetime import datetime
from matplotlib.patches import Ellipse
from sunpy.coordinates import frames
import glob
from plot_aia_lofar import get_beam
import warnings
warnings.simplefilter("ignore")

plt.rcParams['font.size'] = 12 

no_multi = glob.glob("/mnt/murphp30_data/typeIII_int/briggs_comparison/*image.fits")
no_multi.sort()
no_multi[0], no_multi[1] = no_multi[1], no_multi[0]

multi = glob.glob("/mnt/murphp30_data/typeIII_int/briggs_comparison/multiscale/*image.fits")
multi.sort()
multi[0], multi[1] = multi[1], multi[0]

aiafile = "/mnt/murphp30_data/typeIII_int/scripts/AIA20171015.fits"
aiamap = Map(aiafile)
def make_comp(fitsfile):
	briggs= Map(fitsfile)
	briggs = icrs_to_helio.icrs_to_helio(briggs)
	briggs.plot_settings['cmap'] = 'viridis'
	lmax = (briggs.data).max()
	levels = lmax*np.arange(0.5, 1.1, 0.05)
	comp_map = sunpy.map.Map(aiamap, briggs, composite=True)
	comp_map.set_levels(index=1, levels=levels)
	return comp_map


comp_maps_no_multi = [make_comp(file) for file in no_multi]
comp_maps_multi = [make_comp(file) for file in multi]

axlims = [-2500,2500]
fig = plt.figure(figsize=(18,8))
gs = GridSpec(2, 5)

ax10 = fig.add_subplot(gs[1,0])
ax11 = fig.add_subplot(gs[1,1], sharey=ax10)
ax12 = fig.add_subplot(gs[1,2], sharey=ax10)
ax13 = fig.add_subplot(gs[1,3], sharey=ax10)
ax14 = fig.add_subplot(gs[1,4], sharey=ax10)

ax00 = fig.add_subplot(gs[0,0], sharex=ax10)
ax01 = fig.add_subplot(gs[0,1], sharey=ax00)
ax02 = fig.add_subplot(gs[0,2], sharey=ax00)
ax03 = fig.add_subplot(gs[0,3], sharey=ax00)
ax04 = fig.add_subplot(gs[0,4], sharey=ax00)

plt.setp(ax00.get_xticklabels(), visible=False)
plt.setp(ax00.get_xticklines(), visible=False)
for ax in [ax01,ax02,ax03,ax04,ax11,ax12,ax13,ax14]:
	plt.setp(ax.get_xticklabels(), visible=False)
	plt.setp(ax.get_yticklabels(), visible=False)
	plt.setp(ax.get_xticklines(), visible=False)
	plt.setp(ax.get_yticklines(), visible=False)
for ax in [ax00,ax01,ax02,ax03,ax04,ax10,ax11,ax12,ax13,ax14]:
	ax.spines['bottom'].set_color('white')
	ax.spines['top'].set_color('white')
	ax.spines['right'].set_color('white')
	ax.spines['left'].set_color('white')
	ax.set_aspect('equal')
plt.subplots_adjust(hspace=0.0)
plt.subplots_adjust(wspace=0.0)

for comp_map,ax in zip(comp_maps_no_multi,[ax00,ax01,ax02,ax03,ax04]):
	try:
		comp_map.plot(ax,title=' ')
	except: ValueError
	ax.set_xlabel(' ')
	ax.set_ylabel(' ')
	ax.set_xlim(axlims)
	ax.set_ylim(axlims)
	# ax.set_title(' ')
	ax.patch.set_facecolor('black')

try:
	comp_maps_multi[0].plot(ax10,title=' ')
except: ValueError
ax10.set_xlabel(' ')
ax10.set_ylabel(' ')
ax10.set_xlim(axlims)
ax10.set_ylim(axlims)
# ax10.set_title(' ')
ax10.patch.set_facecolor('black')

for comp_map,ax in zip(comp_maps_multi[1:],[ax11,ax12,ax13,ax14]):
	try:
		comp_map.plot(ax,title=' ')
	except: ValueError
	ax.set_xlabel(' ')
	ax.set_ylabel(' ')
	ax.set_xlim(axlims)
	ax.set_ylim(axlims)
	# ax.set_title(' ')
	ax.patch.set_facecolor('black')


for ax, title in zip([ax00,ax01,ax02,ax03,ax04],["Briggs -2", "Briggs -1", "Briggs 0", "Briggs 1", "Briggs 2"]):
	ax.set_title(title)

ax00.set_ylabel("No Multiscale (Solar-Y)")
ax10.set_ylabel("Multiscale (Solar-Y)")
ax10.set_xlabel("Solar-X (arcsec)")
gs.tight_layout(fig, h_pad=-2, w_pad=-2)
plt.savefig("briggs_comparison.png", dpi=400)
plt.show()
