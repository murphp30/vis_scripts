#!/usr/bin/env python

"""
Fit a gauss to a single burst in image space
Pearse Murphy 30/03/20 COVID-19
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sunpy
from sunpy.map import Map
from sunpy.coordinates.frames import Helioprojective
import astropy.units as u
from astropy.coordinates import Angle, SkyCoord, Latitude, Longitude 
from lmfit import Parameters, Model
import icrs_to_helio as icrs_to_helio
#import pdb
import warnings
warnings.filterwarnings("ignore")
def gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
	#create a 2D gaussian with input parameters
	#can't do this because it takes too long, assume it's been done outside function
	#	x = xy.Tx.arcsec
	#	y = xy.Ty.arcsec
	(x, y) = xy
	x0 = float(x0)
	y0 = float(y0)
	a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
	b = ((np.sin(2*theta))/(4*sig_x**2)) - ((np.sin(2*theta))/(4*sig_y**2))
	c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
	g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
	return g#.ravel()

def pix_locs(smap):
	#return real world value of every pixel in smap
	xy_pix = np.indices(smap.data.shape)*u.pix
	xy_mesh = smap.pixel_to_world(xy_pix[0], xy_pix[1])
	return xy_mesh

def make_init_params(smap):

	max_xy = np.where(smap.data == smap.data.max())
	max_pos = smap.pixel_to_world(max_xy[1][0]*u.pix, max_xy[0][0]*u.pix)


	init_params = {"amp":smap.data.max(),
				   "x0":max_pos.Tx.arcsec,
				   "y0":max_pos.Ty.arcsec,
				   "sig_x":Angle(10*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
				   "sig_y":Angle(18*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
				   "theta":0.1,
				   "offset":0}
	return init_params 

def make_params(smap):
	init_params = make_init_params(smap)
	params = Parameters()
	params.add_many(("amp", init_params["amp"], True, 0.5*init_params["amp"], None),
					("x0", init_params["x0"], True, init_params["x0"] - 600, init_params["x0"] + 600),
					("y0", init_params["y0"], True, init_params["y0"] - 600, init_params["y0"] + 600),
					("sig_x", init_params["sig_x"], True, 0, 1.5*init_params["sig_x"]),
					("sig_y", init_params["sig_y"], True, 0, 1.5*init_params["sig_y"]),
					("theta", init_params["theta"], True, 0, np.pi),
					("offset", init_params["offset"], True, smap.data.min(),0.01*smap.data.max() ))
	return params

def rotate_zoom(smap, x0, y0,theta):
	#shift = smap.shift(x0, y0)
	top_right = SkyCoord( x0 + 2000 * u.arcsec, y0 + 2000 * u.arcsec, frame=smap.coordinate_frame)
	bottom_left = SkyCoord( x0 - 2000 * u.arcsec, y0 - 2000 * u.arcsec, frame=smap.coordinate_frame)
	zoom = smap.submap(bottom_left, top_right)
	rot = zoom.rotate(-theta)
	return rot
#loading stuff
#pdb.set_trace()
lofarfile = sys.argv[1]
lofarmap = Map(lofarfile)
lofarmap.plot_settings['cmap'] = 'viridis'
heliomap = icrs_to_helio.icrs_to_helio(lofarmap)
heliomap.plot_settings['cmap'] = 'viridis'


#defining initial params
xy_mesh = pix_locs(heliomap).T
xy_arcsec = [xy_mesh.Tx.arcsec, xy_mesh.Ty.arcsec]
#Fitting stuff
gmodel = Model(gauss_2d)
#init_params = make_init_params(heliomap)
model_gauss = gauss_2d(xy_arcsec,2000,-300,50,
			Angle(9*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
			Angle(19*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
			0.5,10)
#model_gauss = model_gauss.T
noise = 0.06
model_gauss = model_gauss + noise*model_gauss.max()*np.random.normal(size=model_gauss.shape)
model_map = sunpy.map.Map(model_gauss, heliomap.meta)
params = make_params(model_map)
#params = make_params(heliomap)
print("Beginning fit")
gfit = gmodel.fit(model_map.data, params, xy=xy_arcsec)
#gfit = gmodel.fit(heliomap.data, params, xy=xy_arcsec)
#this takes longer than I would like something to do with it not being a np.meshgrid?
#Preparing stuff for a pretty plot
x0 = gfit.params['x0'] * u.arcsec
y0 = gfit.params['y0'] * u.arcsec
theta = Angle(gfit.params['theta'] * u.rad)
gauss_centre = Helioprojective(x0,y0, observer='earth', obstime=heliomap.date)
fit_map = sunpy.map.Map(gfit.best_fit, heliomap.meta)

rot_fit = rotate_zoom(fit_map, x0, y0, theta) #fit_zoom.rotate(-theta)
rot_helio = rotate_zoom(heliomap, x0, y0, theta)#helio_zoom.rotate(-theta)
rot_model = rotate_zoom(model_map, x0, y0, theta) #model_zoom.rotate(-theta)

#zoom_centre = rot_helio.world_to_pixel(gauss_centre)
zoom_centre = rot_model.world_to_pixel(gauss_centre)
#zoom_xy = pix_locs(rot_helio)
zoom_xy = pix_locs(rot_model)
x_cen = int(zoom_centre.x.round().value)
y_cen = int(zoom_centre.y.round().value)
x_1D_model, y_1D_model =  rot_model.data[:,y_cen], rot_model.data[x_cen,:]
x_1D_helio, y_1D_helio =  rot_helio.data[:,y_cen], rot_helio.data[x_cen,:]
x_1D_fit, y_1D_fit =  rot_fit.data[:,y_cen], rot_fit.data[x_cen,:]
zoom_xarr = zoom_xy[:, y_cen]#zoom_xy.Tx[0]
zoom_yarr = zoom_xy[x_cen, :]#zoom_xy.Ty.T[0]
#coord_x = rot_helio.pixel_to_world([x_cen, x_cen]*u.pix, [0,(zoom_xy.shape[0]-1)]*u.pix)
#coord_y = rot_helio.pixel_to_world([0,(zoom_xy.shape[1]-1)]*u.pix, [y_cen, y_cen]*u.pix)
coord_x = rot_model.pixel_to_world([x_cen, x_cen]*u.pix, [0,(zoom_xy.shape[0]-1)]*u.pix)
coord_y = rot_model.pixel_to_world([0,(zoom_xy.shape[1]-1)]*u.pix, [y_cen, y_cen]*u.pix)
#Printing stuff
print(gfit.fit_report())
fwhmx = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_x']) * u.arcsec).arcmin
fwhmy = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_y']) * u.arcsec).arcmin
print(fwhmx, fwhmy)


#Plotting stuff
#heliomap.plot(title="Burst at {} MHz {}".format(str(np.round(heliomap.wavelength.value,3)),heliomap.date.isot))
##model_map.plot(title="Burst at {} MHz {}".format(str(np.round(heliomap.wavelength.value,3)),heliomap.date.isot))
#fit_map.draw_contours(levels=[50]*u.percent, colors='red')
#plt.figure()
##rot_model.plot()
#rot_helio.plot()
#rot_fit.draw_contours(levels=[50]*u.percent, colors='red')
#plt.figure()
##plt.plot(zoom_xarr, x_1D_model)
#plt.plot(zoom_xarr, x_1D_helio)
#plt.plot(zoom_xarr, x_1D_fit)
#plt.figure()
##plt.plot(zoom_yarr, y_1D_model)
#plt.plot(zoom_yarr, y_1D_helio)
#plt.plot(zoom_yarr, y_1D_fit)
fig = plt.figure(figsize = (6, 6))
gs = GridSpec(4,4)
ax0 = fig.add_subplot(gs[0:1,0:3])
ax = fig.add_subplot(gs[1:4,0:3], projection = rot_helio)
ax1 = fig.add_subplot(gs[1:4,3])
#rot_helio.plot(axes=ax, title='')
rot_model.plot(axes=ax, title='')
rot_fit.draw_contours(axes=ax,levels=[50]*u.percent, colors=['red'])
ax.plot_coord(coord_x, '--')
ax.plot_coord(coord_y, '--')
#ax0.plot(zoom_yarr.Ty,y_1D_helio,drawstyle='steps-mid')
ax0.plot(zoom_yarr.Ty,y_1D_model,drawstyle='steps-mid')
ax0.plot(zoom_yarr.Ty,y_1D_fit)
#ax1.plot(x_1D_helio,zoom_xarr.Tx,drawstyle='steps-mid')
ax1.plot(x_1D_model,zoom_xarr.Tx,drawstyle='steps-mid')
ax1.plot(x_1D_fit,zoom_xarr.Tx)

ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax0.set_yticks([])
ax0.set_xticks([])
ax1.set_yticks([])
ax1.set_xticks([])
plt.savefig("gauss_fit_data.png", dpi=400)
plt.show()

