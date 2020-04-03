#!/usr/bin/env python

"""
Fit a gauss to a single burst in image space
Pearse Murphy 30/03/20 COVID-19
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
import sunpy
from sunpy.map import Map
import astropy.units as u
from astropy.coordinates import Angle
from lmfit import Parameters, Model
import icrs_to_helio as icrs_to_helio


def gauss_2d(xy, amp, x0, y0, sig_x, sig_y, theta, offset):
        (x, y) = xy
        x0 = float(x0)
        y0 = float(y0)
        a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
        b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
        c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))
        g = amp*np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))) + offset
        return g#.ravel()

lofarfile = sys.argv[1]
lofarmap = Map(lofarfile)
lofarmap.plot_settings['cmap'] = 'viridis'
heliomap = icrs_to_helio.icrs_to_helio(lofarmap)
heliomap.plot_settings['cmap'] = 'viridis'

x_arr, y_arr = np.arange(heliomap.data.shape[0]), np.arange(heliomap.data.shape[0])
pix_to_world = heliomap.pixel_to_world(x_arr*u.pix, y_arr*u.pix)
x_arr = pix_to_world.Tx.arcsec
y_arr = pix_to_world.Ty.arcsec

max_xy = np.where(heliomap.data == heliomap.data.max())



init_params = {"amp":heliomap.data.max(),
               "x0":x_arr[max_xy[1][0]],
               "y0":y_arr[max_xy[0][0]],
               "sig_x":Angle(10*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
               "sig_y":Angle(18*u.arcmin).arcsec/(2 * np.sqrt(2*np.log(2))),
               "theta":0,
               "offset":heliomap.data.min()}

# x and y position are swapped to what you would expect and I don't know why.

gmodel = Model(gauss_2d)
params = Parameters()
params.add_many(("amp", init_params["amp"], True, 0.5*init_params["amp"], None),
                ("x0", init_params["x0"], True, init_params["x0"] - 600, init_params["x0"] + 600),
                ("y0", init_params["y0"], True, init_params["y0"] - 600, init_params["y0"] + 600),
                ("sig_x", init_params["sig_x"], True, 0, 2*init_params["sig_x"]),
                ("sig_y", init_params["sig_y"], True, 0, 2*init_params["sig_y"]),
                ("theta", init_params["theta"], True, 0, np.pi),
                ("offset", 1.5*init_params["offset"], True, init_params["offset"], 0))

xy_mesh = np.meshgrid(x_arr, y_arr)

gfit = gmodel.fit(heliomap.data, params, xy=xy_mesh)

fit_map = sunpy.map.Map(gfit.best_fit, heliomap.meta)
heliomap.plot(title="Burst at {} MHz {}".format(str(np.round(heliomap.wavelength.value,3)),heliomap.date.isot))
heliomap.draw_limb()
heliomap.draw_grid()
fit_map.draw_contours(levels=[50]*u.percent, colors='red')
#comp_map = sunpy.map.Map(heliomap, fit_map, composite = True)
#levels = [0.5 * gfit.best_fit.max()]
#comp_map.set_levels(index=1, levels=levels)
## comp_map.set_plot_settings(index=1, colors='red')
## heliomap.plot()
#fig = plt.figure()
##ax = fig.add_subplot(111, projection=heliomap)
#comp_map.plot(origin='lower')
#comp_map.draw_limb(0)
#comp_map.draw_grid(0)
# plt.colorbar()
# gfit.params.pretty_print()
print(gfit.fit_report())
fwhmx = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_x']) * u.arcsec).arcmin
fwhmy = Angle((2*np.sqrt(2*np.log(2))*gfit.params['sig_y']) * u.arcsec).arcmin

print(fwhmx, fwhmy)
# plt.figure()
# plt.imshow(gfit.init_fit, origin="lower", extent = [x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]])
# plt.colorbar()
# plt.figure()
# plt.imshow(gfit.best_fit, origin = "lower", vmin = np.min(heliomap.data), vmax = np.max(heliomap.data),
# 			extent = [x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]])
# plt.colorbar()
plt.savefig("SB076_fit_overlay.png",dpi=400)
plt.show()
