#!/usr/bin/env python
import sys
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
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

mpl.rcParams.update({'font.size': 14})

def get_beam(fname):
   
	solar_PA = sunpy.coordinates.sun.P(sunpy.time.parse_time(fits.open(fname)[0].header['DATE-OBS'])).degree
	b_maj =  fits.open(fname)[0].header['BMAJ']
	b_min  = fits.open(fname)[0].header['BMIN']
	b_ang = 90-(fits.open(fname)[0].header['BPA']-solar_PA) # should consider the beam for the data
	return [b_maj, b_min, b_ang] # Output in degrees


def plot_aia_lofar(aiafile, lofarfile, imarcsec=1500, axpos=[0.12, 0.12, 0.8, 0.8], 
				writepng=False, axlims=[-500,4596], beampos=(5500, -1500)):

	aiamap = Map(aiafile)
	lofarmap = Map(lofarfile)

	if not plt.fignum_exists(0): 
		fig = plt.figure(0, figsize=(10,10))
	else:
		fig = plt.figure(0)	

	
	# aiamap.plot()
	# aiamap.draw_grid(axes=ax)

	#-----------------------------------------#
	#	Convert LOFAR map to helioprojective.
	#	Add contours.
	lofarmap = icrs_to_helio.icrs_to_helio(lofarmap)
	lofarmap.plot_settings['cmap'] = 'viridis'
	lofarwcs = wcs.WCS(lofarmap.meta)
	lofarwcs.heliographic_observer = 'earth'#aiamap.observer_coordinate
	lmax = (lofarmap.data).max()
	if lmax<1e4: lmax=1e4 #Nothing less than 1 SFU/beam
	levels = lmax*np.arange(0.5, 1.1, 0.05)
	# ax.contourf(lofarmap.data, 
	# 		transform=ax.get_transform(lofarwcs), 
	# 		levels=levels, 
	# 		alpha=0.6, 
	# 		antialiased=True, 
	# 		cmap=plt.get_cmap('GnBu_r'))
	comp_map = sunpy.map.Map(aiamap, lofarmap, composite=True)
	comp_map.set_levels(index=1, levels=levels)
	ax = fig.add_subplot(111)#add_axes(axpos)#, projection=aiamap)
	comp_map.plot(origin='lower')
	# comp_map.draw_grid()
	ax.set_xlim(axlims)
	ax.set_ylim(axlims)
	ax.set_title(' ')
	ax.patch.set_facecolor('black')

	#levels = lmax*np.arange(0.5, 1.1, 0.05)
	#ax.contour(lofarmap.data, transform=ax.get_transform(lofarwcs), 
	#		levels=levels, alpha=0.8, antialiased=True, cmap=plt.get_cmap('RdPu_r'), linewidths=2)

	lofar_freq = round(lofarmap.meta['wavelnth'],2)
	frq_str = str(lofar_freq)+' MHz'
	aia_tim_fmt = aiamap.meta['date-obs']
	aia_dt = datetime.strptime(aia_tim_fmt, '%Y-%m-%dT%H:%M:%S.%f')
	tim_file = aia_dt.strftime('%Y%m%d_%H%M%S')
	aia_filter = str(aiamap.meta['wavelnth']/10)

	image_name = lofarfile[-19:-9]+'.png'#'aia_lofar_'+tim_file+'.png'
	plt.text(0.01, 0.05, 'LOFAR '+frq_str, color='mediumaquamarine', transform=ax.transAxes) 
	plt.text(0.01, 0.01, 'AIA '+aia_filter+'nm '+aia_tim_fmt, color='gold', transform=ax.transAxes) 

	b_major, b_min, b_angle = get_beam(lofarfile)
	beam = Ellipse(beampos, b_major*3600., b_min*3600., b_angle, color='blue', fill=False, linewidth=2)
	ax.add_artist(beam)

	#pdb.set_trace()
	if writepng:
		plt.savefig('./'+image_name)
		print('Writing %s' %(image_name))
		# plt.show()


if __name__ == "__main__":

	#-----------------------------------------#
	#
	#				Read AIA
	#
	#aiafile  = sys.argv[1]
	lofarfile  = sys.argv[1]
	aiafile = sys.argv[2] #'/Users/murphp30/Documents/Postgrad/my_papers/20151017/AIA20171015.fits'#'/Users/eoincarley/Data/2013_may_31/AIA/aia.lev1.171A_2013-05-31T11_11_35.34Z.image_lev1.fits'
	# lofarfile = '/Users/murphp30/mnt/murphp30_data/typeIII_int/briggs_comparison/SB076_b-1-image.fits' #'./wsclean-SB076-b_0-image.fits'#'/Users/murphp30/mnt/murphp30_data/typeIII_int/SB076_auto_b0-image.fits'#'./wsclean-SB076-b_0-image.fits'#gain_corrections/bm1/multiscale/wsclean-t0012-image.fits'#'/Users/eoincarley/Data/2013_may_31/lofar/fits_SB008_58MHz/L141641_SAP000_44MHz-t0132-image.fits'

	plot_aia_lofar(aiafile, lofarfile, writepng=False, axlims=[-2500,2500], beampos=(1500,-1500))

plt.show()

