import os
import astropy.constants as const
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sunpy.map

import pfsspy
from pfsspy import coords
from pfsspy import tracing

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


[[br, header]] = sunpy.io.fits.read('151017t1104gong.fits')
br = br - np.mean(br)
br = np.roll(br, header['CRVAL1'] + 180, axis=1)

nrho = 60
rss = 2.5
input = pfsspy.Input(br, nrho, rss)
fig, ax = plt.subplots()
mesh = input.plot_input(ax)
fig.colorbar(mesh)
ax.set_title('Input field')


output = pfsspy.pfss(input)
output.plot_pil(ax)

fig, ax = plt.subplots()
mesh = output.plot_source_surface(ax)
fig.colorbar(mesh)
output.plot_pil(ax)
ax.set_title('Source surface magnetic field')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

tracer = tracing.PythonTracer()
# Loop through 16 values in theta and 16 values in phi
r = 1.01
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2 * np.pi, 50)
theta, phi = np.meshgrid(theta, phi)
theta, phi = theta.ravel(), phi.ravel()

seeds = np.array(coords.sph2cart(r, theta, phi)).T

field_lines = tracer.trace(seeds, output)

for field_line in field_lines.open_field_lines:
    #color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    color = {-1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
    ax.plot(field_line.coords.x / const.R_sun,
            field_line.coords.y / const.R_sun,
            field_line.coords.z / const.R_sun,
            color=color, linewidth=1)

ax.set_title('PFSS solution')

plt.show() 
