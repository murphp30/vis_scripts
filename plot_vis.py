#!/usr/bin/env python

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime, timedelta
from scipy import interpolate
import scipy.optimize as opt
from astropy.constants import c
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from lmfit import Model, Parameters, Minimizer,minimize
import corner
import emcee
from multiprocessing import Pool
import time
import warnings
from sunpy.coordinates import sun
import argparse
from plot_bf import get_data

parser = argparse.ArgumentParser()
parser.add_argument('--vis_file', dest='vis_file', help='Input visibility file. \
	Must be npz file of the same format as that created by vis_to_npy.py', default='SB076MS_data.npz')
parser.add_argument('--bf_file', dest='bf_file', help='Input HDF5 file containing beamformed data', default='L401005_SAP000_B000_S0_P000_bf.h5')
args = parser.parse_args()
vis_file = args.vis_file
bf_file = args.bf_file
warnings.simplefilter("ignore", category=FutureWarning)

t0 = time.time()

def FWHM(sig):
	fwhm = (2*np.sqrt(2*np.log(2))*sig)
	fwhm = Angle(fwhm*u.rad).arcmin
	return fwhm

def rotate_coords(u,v, theta):
	# rotate coordinates u v by theta
	u_p =  u*np.cos(theta) - v*np.sin(theta)
	v_p = u*np.sin(theta) + v*np.cos(theta)
	return u_p,v_p

def gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	u_p,v_p  =  rotate_coords(u,v,theta)#u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	#x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u*x0+v*y0)) \
	* np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))
	
	return V

def residual(pars, u,v, data, weights=None, ngauss=1, size=True):
	parvals = pars.valuesdict()
	if ngauss == 1:
		I0 = parvals['I0']
		sig_x = parvals['sig_x']
		sig_y = parvals['sig_y']
		theta = parvals['theta']
		if size:
			x0 = 0
			y0 = 0
		
		else:
			x0 = parvals['x0']
			y0 = parvals['y0']
		
		model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta)

	elif ngauss == 2:
		I0 = parvals['I0']
		sig_x0 = parvals['sig_x0']
		sig_y0 = parvals['sig_y0']
		theta0 = parvals['theta0']
		I1 = parvals['I1']
		sig_x1 = parvals['sig_x1']
		sig_y1 = parvals['sig_y1']
		theta1 = parvals['theta1']
		if size:
			x0 = 0
			y0 = 0
			x1 = 0
			y1 = 0		
		else:
			x0 = parvals['x0']
			y0 = parvals['y0']
			x1 = parvals['x1']
			y1 = parvals['y1']
		model = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0) + gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1)
	else:
		print("Must have max 2 gauss (for now)")
		return
	if size:
		if weights is None:
			resid = abs(model) - abs(data) #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
		else:
			resid = (abs(model)-abs(data))*weights#np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
	else:
		if weights is None:
			resid = (model.real - data.real)**2 + (model.imag-data.imag)**2 #np.angle(model) - np.angle(data)
		else:
			resid = ((model.real - data.real)**2 + (model.imag-data.imag)**2)*weights#(np.angle(model)-np.angle(data))*weights		
	return resid

def lnlike(pars, u,v,vis,weights=None, size=True):
	try:
		x0,y0 = pars
	except ValueError:
		I0,sig_x,sig_y,theta = pars
	
	if size:
		x0 = 0
		y0 = 0
	else:
		size_fit = [*fit.params.valuesdict().values()]
		I0, sig_x, sig_y, theta = size_fit[0], *size_fit[3:]
	u_p,v_p  =  rotate_coords(u,v,theta)
	model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta)
	if weights is None:
		inv_sigma2 = np.ones(len(vis))
	else:
		inv_sigma2 = weights 
	

	if size:
		diff = abs(vis) - abs(model)
	else:
		diff = np.imag(vis) - np.imag(model)
	return -0.5*(np.sum(diff**2*inv_sigma2 - np.log(2*np.pi*inv_sigma2)))

def lnprior(pars,vis,size=True):
	try:
		x0,y0 = pars
	except ValueError:
		I0,sig_x,sig_y,theta = pars
	if size:
		x0 = 0
		y0 = 0
	else:
		size_fit = [*fit.params.valuesdict().values()]
		I0, sig_x, sig_y = size_fit[0], *size_fit[3:-1]
	
	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	if size:
		if np.max(abs(vis)) < I0 < 10*np.max(abs(vis)) and sig_stria < sig_x < 1*sig_sun and sig_stria < sig_y < 1*sig_sun \
		and -np.pi < theta < 0:
			return 0.0
		return -np.inf

	else:
		if -2*sun_diam_rad < x0 < 1*sun_diam_rad and -2*sun_diam_rad < y0 < 1*sun_diam_rad:
			return 0.0
		return -np.inf

def lnprob(pars, u,v,vis,weights=None, size=True):
	lp = lnprior(pars,vis,size)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(pars, u,v,vis,weights, size)

def two_gauss_V(u,v, I0,x0,y0,sig_x0,sig_y0,theta0,I1,x1,y1,sig_x1,sig_y1,theta1):
	
	V0 = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0)
	V1 = gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1)

	V = V0 + V1
	return V

def two_gauss_I(x,y, I0,x0,y0,sig_x0,sig_y0,theta0,I1,x1,y1,sig_x1,sig_y1,theta1):
	
	I0 = gauss_I(x,y,I0,x0,y0,sig_x0,sig_y0,theta0)
	I1 = gauss_I(x,y,I1,x1,y1,sig_x1,sig_y1,theta1)

	I = I0 + I1
	return I

def gauss_I(x,y,I0,x0,y0,sig_x,sig_y,theta):
	
	# a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
	# b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
	# c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))

	x_p = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)
	y_p = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
	
	I = (I0/(2*np.pi*sig_x*sig_y)) * np.exp(-( (x_p**2/(2*sig_x**2)) + (y_p**2/(2*sig_y**2)) ))
	#(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

	return I

class LOFAR_vis:
	"""
	A class that contains a LOFAR measurment set 
	and the various ways it's split into useful things
	requires a npz file name created with vis_to_npy.py
	and a time sample number/timestamp
	"""

	def __init__(self, fname, t):
		self.fname = fname
		self.t = t
		
		self.load_vis = np.load(self.fname)
		self.freq = float(self.load_vis["freq"])
		phs_dir = self.load_vis["phs_dir"]
		self.phs_dir =  SkyCoord(phs_dir[0], phs_dir[1], unit="rad")

		self.delt = float(self.load_vis["dt"])
		self.delf = float(self.load_vis["df"])
		self.wlen = c.value/self.freq
		self.obsstart = epoch_start + timedelta(seconds=self.load_vis["times"][0][0])
		self.obsend = epoch_start + timedelta(seconds=self.load_vis["times"][-1][0])

		self.time = epoch_start + timedelta(seconds=self.load_vis["times"][self.t][0])

		self.sun_dir =  sun.sky_position(self.time,False)
		self.solar_ra_offset = (self.phs_dir.ra-self.sun_dir[0])
		self.solar_dec_offset = (self.phs_dir.dec-self.sun_dir[1])
		self.solar_rad = sun.angular_radius(self.time)

	def vis_df(self):
		ant0 = self.load_vis["ant0"]
		ant1 = self.load_vis["ant1"]
		#auto_corrs = np.where(ant0==ant1)[0]
		cross_corrs = np.where(ant0!=ant1)[0]

		uvws = self.load_vis["uvws"]/self.wlen
		times = self.load_vis["times"]
		times = epoch_start + timedelta(seconds=1)*times
		data = self.load_vis["data"]
		vis = self.load_vis["vis"]
		weights = self.load_vis["weights"]

		data = data[0,:] + data[3,:]
		V = vis[0,:] + vis[3,:]
		vis_err = np.sqrt(1/weights*abs(vis))
		vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
		weights = (weights[0,:]*abs(vis)[0,:] + weights[3,:]*abs(vis)[3,:])		
		
		d_cross = {"u":uvws[0,self.t,:][cross_corrs],"v":uvws[1,self.t,:][cross_corrs],"w":uvws[2,self.t,:][cross_corrs], 
		"times":times[self.t,cross_corrs], "raw":data[self.t, cross_corrs], "vis":V[self.t,cross_corrs], "vis_err":vis_err[self.t,cross_corrs],
		"weight":weights[self.t,cross_corrs]}
		df_cross = pd.DataFrame(data=d_cross)

		uv_dist = np.sqrt(df_cross.u**2 + df_cross.v**2)
		ang_scales = Angle((1/uv_dist)*u.rad)

		bg = np.where(ang_scales.arcmin < 2 )[0]
		bg_vis = df_cross.vis[bg]
		bg_mean = np.mean(bg_vis)		

		df_cross = df_cross.assign(uv_dist = uv_dist)
		df_cross = df_cross.assign(ang_scales = ang_scales.arcmin)
		df_cross = df_cross.assign(bg_vis = (df_cross.vis - bg_mean))

		return df_cross

	def queit_sun_df(self):
		ant0 = self.load_vis["ant0"]
		ant1 = self.load_vis["ant1"]
		#auto_corrs = np.where(ant0==ant1)[0]
		cross_corrs = np.where(ant0!=ant1)[0]

		q_t = 1199 #time index before burst, first 10 minutes of data is queit sun ~ 1199 time samples
		uvws = self.load_vis["uvws"][:,:q_t,:]/self.wlen
		data = self.load_vis["data"][:,:q_t,:]
		vis = self.load_vis["vis"][:,:q_t,:]
		weights = self.load_vis["weights"][:,:q_t,:]

		data = data[0,:] + data[3,:]
		V = vis[0,:] + vis[3,:]
		vis_err = np.sqrt(1/weights*abs(vis))
		vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
		weights = (weights[0,:]*abs(vis)[0,:] + weights[3,:]*abs(vis)[3,:])		
		
		uvws = np.mean(uvws, axis=1)
		V = np.mean(V, axis=0)
		data = np.mean(data, axis=0)
		vis_err = np.mean(vis_err, axis=0)
		weights = np.mean(weights, axis=0)


		d_cross = {"u":uvws[0,:][cross_corrs],"v":uvws[1,:][cross_corrs],"w":uvws[2,:][cross_corrs], "raw":data[cross_corrs],
		"vis":V[cross_corrs], "vis_err":vis_err[cross_corrs], "weight":weights[cross_corrs]}
		df_cross = pd.DataFrame(data=d_cross)

		uv_dist = np.sqrt(df_cross.u**2 + df_cross.v**2)
		ang_scales = Angle((1/uv_dist)*u.rad)

		bg = np.where(ang_scales.arcmin < 2 )[0]
		bg_vis = df_cross.vis[bg]
		bg_mean = np.mean(bg_vis)		

		df_cross = df_cross.assign(ang_scales = ang_scales.arcmin)
		df_cross = df_cross.assign(bg_vis = (df_cross.vis - bg_mean))

		return df_cross

"""
------
"""

q_t = 1199 #time index before burst, first 10 minutes of data is queit sun ~ 1199 time samples

SB = int(vis_file.split("SB")[-1][:3])
epoch_start = datetime(1858,11,17) #MJD

sun_diam_rad = Angle(0.5*u.deg).rad
sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

stria_oom = Angle(0.1*u.arcmin).rad
sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

scatter_diam = Angle(10*u.arcmin).rad
sig_scatter = scatter_diam/(2*np.sqrt(2*np.log(2)))



x_guess = Angle(11*u.arcmin).rad
y_guess = Angle(12*u.arcmin).rad
sig_x_guess = x_guess/(2*np.sqrt(2*np.log(2)))
sig_y_guess = y_guess/(2*np.sqrt(2*np.log(2)))

x1_guess = Angle(5*u.arcmin).rad
y1_guess = Angle(5*u.arcmin).rad
sig_x1_guess = x1_guess/(2*np.sqrt(2*np.log(2)))
sig_y1_guess = y1_guess/(2*np.sqrt(2*np.log(2)))

vis0 = LOFAR_vis(vis_file, 0)
q_sun = vis0.queit_sun_df()
arr_size = 5000
u_arr = np.arange(q_sun.u.min(),q_sun.u.max(),(q_sun.u.max()-q_sun.u.min())/arr_size )
v_arr = np.arange(q_sun.v.min(),q_sun.v.max(),(q_sun.v.max()-q_sun.v.min())/arr_size )
uv_mesh = np.meshgrid(u_arr,v_arr) 
x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
xy_mesh = np.meshgrid(x_arr,y_arr) 

bf_data, bf_freq, bf_tarr = get_data(bf_file, vis0.obsstart, vis0.obsend )
bf_delt = bf_tarr[1] - bf_tarr[0]
bf_delf = bf_freq[1] - bf_freq[0]


def parallel_fit(i):
	save = False
	vis = LOFAR_vis(vis_file, i)
	burst = vis.vis_df()

	ngauss = 1

	params = Parameters()
	fit_vis = burst.bg_vis - q_sun.bg_vis
	fit_weight = abs(burst.weight - q_sun.weight)
	if ngauss == 2:
		params.add_many(('I0',np.pi*np.max(abs(fit_vis)),True,0,abs(np.max(fit_vis))*10), 
			('x0',-0.7*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad),
			('y0',-0.5*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x0',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y0',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta0',np.pi/3,True,0,np.pi),
			('I1',np.pi*np.max(abs(fit_vis))/2,True,0,abs(np.max(fit_vis))*10), 
			('x1',-0.7*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad),
			('y1',-0.5*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x1',sig_x1_guess,True,sig_stria,1.5*sig_sun),
			('sig_y1',sig_y1_guess,True,sig_stria,1.5*sig_sun), 
			('theta1',np.pi/3,True,0,np.pi))
	elif ngauss == 1:
		params.add_many(('I0',2*np.pi*np.max(abs(fit_vis)),True,0,abs(np.max(fit_vis))*10), 
			('x0',-0.7*sun_diam_rad,False,-2*sun_diam_rad,-0.25*sun_diam_rad),
			('y0',-0.5*sun_diam_rad,False,-2*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta',np.pi/3,True,0, np.pi))


	fit = minimize(residual, params, method="emcee", args=(burst.u, burst.v, fit_vis, fit_weight , ngauss, True))
	print("Fitting", i-q_t)

	if ngauss == 2:
		fit.params["I0"].vary = False
		fit.params["x0"].vary = True
		fit.params["y0"].vary = True
		fit.params["sig_x0"].vary = False
		fit.params["sig_y0"].vary = False
		fit.params["theta0"].vary = False
		fit.params["I1"].vary = False
		fit.params["x1"].vary = True
		fit.params["y1"].vary = True
		fit.params["sig_x1"].vary = False
		fit.params["sig_y1"].vary = False
		fit.params["theta1"].vary = False
	elif ngauss == 1:
		fit.params["I0"].vary = False
		fit.params["x0"].vary = True
		fit.params["y0"].vary = True
		fit.params["sig_x"].vary = False
		fit.params["sig_y"].vary = False
		fit.params["theta"].vary = False


	fit_pos = minimize(residual, fit.params, method="emcee", args=(burst.u, burst.v, fit_vis,fit_weight ,ngauss,False))
	val_dict = fit_pos.params.valuesdict()

	if ngauss == 2:
		g_fit = two_gauss_V(burst.u, burst.v, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x0'], val_dict['sig_y0'], val_dict['theta0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
			val_dict['sig_x1'], val_dict['sig_y1'], val_dict['theta1'])

		# gm_fit = two_gauss_V(uv_mesh[0], uv_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
		# 	val_dict['sig_x0'], val_dict['sig_y0'], -val_dict['theta0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
		# 	val_dict['sig_x1'], val_dict['sig_y1'], -val_dict['theta1'])

		I_fit = two_gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x0'], val_dict['sig_y0'], -val_dict['theta0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
			val_dict['sig_x1'], val_dict['sig_y1'], -val_dict['theta1'])

	elif ngauss == 1:
		g_fit = gauss_2D(burst.u, burst.v, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'])

		# gm_fit = gauss_2D(uv_mesh[0], uv_mesh[1], val_dict['I0'], mc_pars[1,0], mc_pars[1,1], 
		# 	val_dict['sig_x'], val_dict['sig_y'], -val_dict['theta'])

		I_fit = gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], -val_dict['theta'])

	plt.figure()
	plt.plot(burst.ang_scales, abs(fit_vis),'o')
	plt.plot(burst.ang_scales, abs(g_fit),'r+') 
	plt.xlabel("Angular Scale (arcminute)")
	plt.ylabel("Visibility (AU)")
	plt.title("Visibility vs angular scale {}".format(vis.time.isoformat()))
	plt.xscale('log')
	plt.tight_layout()
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/SB{0}/vis_ang_scale_t{1}_1gauss.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		plt.close()
	# plt.figure()
	# plt.scatter(burst.u, burst.v, c=np.real(fit_vis)) 
	# plt.imshow(np.real(gm_fit), aspect='auto', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]],
	#  vmin=np.min(np.real(fit_vis)), vmax=np.max(np.real(fit_vis))) 
	fig, ax = plt.subplots()
	im = ax.imshow(I_fit, aspect='equal', origin='lower', 
		extent=[Angle(x_arr[0],unit='rad').arcsec, Angle(x_arr[-1],unit='rad').arcsec,
		Angle(y_arr[0],unit='rad').arcsec, Angle(y_arr[-1],unit='rad').arcsec])
	s = Circle((vis.solar_ra_offset.arcsec,vis.solar_dec_offset.arcsec),vis.solar_rad.arcsec, color='r', fill=False)
	ax.add_patch(s)
	plt.xlabel("X (arcsecond)")
	plt.ylabel("Y (arcsecond)")
	fig.colorbar(im)
	plt.title("Recreated Image {}".format(vis.time.isoformat()))
	plt.tight_layout()
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/SB{0}/im_recreate_t{1}_1gauss.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		plt.close()
	return fit_pos
	#fig, ax = plt.subplots()
	#ax.imshow(abs(gm_fit), aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
	# s = Circle((0,0),1/vis.solar_rad.rad, color='r', fill=False)
	# ax.add_patch(s)

	
# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)
# with Pool() as p_fit:
# 	fits = p_fit.map(parallel_fit, range(q_t, q_t+79))
parallel_fit(q_t+25)
t_run = time.time()-t0
print("Time to run:", t_run)
plt.show()

