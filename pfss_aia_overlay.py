from datetime import datetime
import os

import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sunpy.map
import sunpy.io.fits

import pfsspy
import pfsspy.coords as coords
import pfsspy.tracing as tracing
if not os.path.exists('171015t1304gong.fits') and not os.path.exists('171015t1304gong.fits.gz'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://gong2.nso.edu/oQR/zqs/201510/mrzqs151017/mrzqs151017t1104c2169_130.fits.gz',
        '151017t1104gong.fits.gz')

if not os.path.exists('151017t1104gong.fits'):
    import gzip
    with gzip.open('151017t1104gong.fits.gz', 'rb') as f:
        with open('151017t1104gong.fits', 'wb') as g:
            g.write(f.read())


if not os.path.exists('AIA20171015.fits'):
    import urllib.request
    urllib.request.urlretrieve(
        'http://jsoc2.stanford.edu/data/aia/synoptic/2017/10/15/H1300/AIA20171015_1322_0171.fits',
        'AIA20171015.fits')

aia = sunpy.map.Map('AIA20171015.fits')
dtime = aia.date

[[br, header]] = sunpy.io.fits.read('151017t1104gong.fits')
br = br - np.mean(br)
br = np.roll(br, header['CRVAL1'] + 180, axis=1)

nrho = 60
rss = 2.5

input = pfsspy.Input(br, nrho, rss, dtime=dtime)

#fig, ax = plt.subplots()
#mesh = input.plot_input(ax)
#fig.colorbar(mesh)
#ax.set_title('Input field')

#plt.figure()
#ax = plt.subplot(1, 1, 1, projection=aia)
#aia.plot(ax)

s, phi = np.meshgrid(np.linspace(-0.5, 0.5, 10),
                     np.deg2rad(np.linspace(85, 145, 10)))

fig, ax = plt.subplots()
mesh = input.plot_input(ax)
fig.colorbar(mesh)
ax.scatter(np.rad2deg(phi), s, color='k', s=1)

ax.set_xlim(80, 150)
ax.set_ylim(-0.5, 0.5)
ax.set_title('Field line footpoints')


output = pfsspy.pfss(input)
tracer = tracing.PythonTracer(atol=1e-8)
flines = []
for s, phi in zip(s.ravel(), phi.ravel()):
    x0 = np.array(pfsspy.coords.strum2cart(0.01, s, phi))
    flines += tracer.trace(x0, output)
fig, ax = plt.subplots()
mesh = input.plot_input(ax)
for fline in flines:
    fline.coords.representation_type = 'spherical'
    ax.plot(fline.coords.lon / u.deg, np.sin(fline.coords.lat),
            color='black', linewidth=1)

ax.set_xlim(80, 150)
ax.set_ylim(-0.5, 0.5)
ax.set_title('Photospheric field and traced field lines')
fig = plt.figure()
ax = plt.subplot(1, 1, 1, projection=aia)
transform = ax.get_transform('world')
aia.plot(ax)
for fline in flines:
    coords = fline.coords.transform_to(aia.coordinate_frame)
    ax.plot_coord(coords, alpha=0.8, linewidth=1, color='red')

ax.patch.set_facecolor('black')
# ax.set_xlim(500, 900)
# ax.set_ylim(400, 800)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# r = 1.01
# theta = np.linspace(0, np.pi/2, 16)
# phi = np.linspace(0, np.pi, 16)
# theta, phi = np.meshgrid(theta, phi)
# theta, phi = theta.ravel(), phi.ravel()

# seeds = np.array(pfsspy.coords.sph2cart(r, theta, phi)).T

# field_lines = tracer.trace(seeds, output)
for field_line in flines:
    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    ax.plot(field_line.coords.cartesian.x / const.R_sun,
            field_line.coords.cartesian.y / const.R_sun,
            field_line.coords.cartesian.z / const.R_sun,
            color=color, linewidth=1)

ax.set_title('PFSS solution')
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.set_xlim(-2.5, 2.5)

plt.show()
