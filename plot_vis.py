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

warnings.simplefilter("ignore", category=FutureWarning)

t0 = time.time()
def phase(Z):
	theta = np.arctan(Z.imag/Z.real)
	theta[np.where(Z.real == 0)] = 0
	return theta

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
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
	* np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))
	
	return V

def residual_abs(pars, u,v, data, weights=None):
	parvals = pars.valuesdict()
	I0 = parvals['I0']
	x0 = 0#parvals['x0']
	y0 = 0#parvals['y0']
	sig_x = parvals['sig_x']
	sig_y = parvals['sig_y']
	theta = parvals['theta']

	model = abs(gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta))
	
	if weights is None:
		resid = model - abs(data) #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
	else:
		resid = (model - abs(data))*weights#np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
	return resid#np.sqrt(resid.real**2 +resid.imag**2)

def residual_pos(pars, u,v, data, weights=None):
	parvals = pars.valuesdict()
	I0 = parvals['I0']
	x0 = parvals['x0']
	y0 = parvals['y0']
	sig_x = parvals['sig_x']
	sig_y = parvals['sig_y']
	theta = parvals['theta']

	model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta)
	
	if weights is None:
		resid = model - data #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
	else:
		resid = (model - data)*weights#np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
	return np.sqrt(resid.real**2 +resid.imag**2)

def lnlike(pars, u,v,vis,weights):
	I0,x0,y0,sig_x,sig_y,theta = pars
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	model = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
		    * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(pars,vis):
	I0,x0,y0,sig_x,sig_y,theta = pars 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun
	sig_y_guess = sig_sun

	if np.min(vis) < I0 < 10*np.max(vis) and -2*sun_diam_rad < x0 < 2*sun_diam_rad and \
 	-2*sun_diam_rad < y0 < 2*sun_diam_rad and sig_stria < sig_x < 2*sig_sun and sig_stria < sig_y < 2*sig_sun \
	and -np.pi/2 < theta < np.pi/2:
		return 0.0
	return -np.inf

def lnprob(pars, u,v,vis,weights):
	lp = lnprior(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(pars, u,v,vis,weights)

def lnlike_real(pars, u,v,vis,weights):
	x0, y0 = pars
	I0, sig_x, sig_y, theta = [*mc_pars[1]]	
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)
	model = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
		    * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))
	inv_sigma2 = weights 
	# phs_model = np.arctan(np.imag(model)/np.real(model))
	# phs_vis = np.arctan(np.imag(vis)/np.real(vis))
	return -0.5*(np.sum((np.real(vis) - np.real(model))**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_real(pars,vis):
	x0, y0 = pars 
	
	sun_diam_rad = Angle(0.5*u.deg).rad

	if -2*sun_diam_rad < x0 < -0.25*sun_diam_rad and \
 	-2*sun_diam_rad < y0 < -0.25*sun_diam_rad:
		return 0.0
	return -np.inf

def lnprob_real(pars, u,v,vis,weights):
	lp = lnprior_real(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real(pars, u,v,vis,weights)

def lnlike_abs(pars, u,v,vis,weights):
	I0,sig_x,sig_y,theta = pars
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	model = abs((I0/(2*np.pi)) * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)))
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_abs(pars,vis):
	I0,sig_x,sig_y,theta = pars 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun
	sig_y_guess = sig_sun

	if np.min(abs(vis)) < I0 < 10*np.max(abs(vis)) and sig_stria < sig_x < 2*sig_sun and sig_stria < sig_y < 2*sig_sun \
	and 0 < theta < np.pi/2:
		return 0.0
	return -np.inf

def lnprob_abs(pars, u,v,vis,weights):
	lp = lnprior_abs(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_abs(pars, u,v,vis,weights)

def lnlike_abs2(pars, u,v,vis,weights):
	I0,sig_x0,sig_y0,theta0,I1,sig_x1,sig_y1,theta1 = pars
	u_p0,v_p0  =  u*np.cos(theta0)+v*np.sin(theta0),-u*np.sin(theta0)+v*np.cos(theta0)
	u_p1,v_p1  =  u*np.cos(theta1)+v*np.sin(theta1),-u*np.sin(theta1)+v*np.cos(theta1)
	model0 = (I0/(2*np.pi)) * np.exp(-(((sig_x0**2)*((2*np.pi*u_p0)**2))/2) - (((sig_y0**2)*((2*np.pi*v_p0)**2))/2))
	model1 = (I1/(2*np.pi)) * np.exp(-(((sig_x1**2)*((2*np.pi*u_p1)**2))/2) - (((sig_y1**2)*((2*np.pi*v_p1)**2))/2))
	model = abs(model0 + model1)
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_abs2(pars,vis):
	I0,sig_x0,sig_y0,theta0,I1,sig_x1,sig_y1,theta1 = pars 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun
	sig_y_guess = sig_sun

	if np.min(vis) < I0 < 10*np.max(vis) and sig_stria < sig_x0 < 2*sig_sun \
	and sig_stria < sig_y0 < 2*sig_sun and 0 < theta0 < np.pi/2 \
	and np.min(vis) < I1 < 10*np.max(vis) and sig_stria < sig_x1 < 2*sig_sun \
	and sig_stria < sig_y1 < 2*sig_sun and 0 < theta1 < np.pi/2:
		return 0.0
	return -np.inf

def lnprob_abs2(pars, u,v,vis,weights):
	lp = lnprior_abs2(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_abs2(pars, u,v,vis,weights)


def lnlike_real2(pars, u,v,vis,weights):
	x0, y0, x1, y1 = pars
	I0,sig_x0,sig_y0,theta0,I1,sig_x1,sig_y1,theta1 = [*mc_pars[1]]	
	u_p0,v_p0  =  u*np.cos(theta0)+v*np.sin(theta0),-u*np.sin(theta0)+v*np.cos(theta0)
	u_p1,v_p1  =  u*np.cos(theta1)+v*np.sin(theta1),-u*np.sin(theta1)+v*np.cos(theta1)
	x0_p, y0_p = rotate_coords(x0,y0,theta0)
	x1_p, y1_p = rotate_coords(x1,y1,theta1)
	model0 = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p0*x0_p+v_p0*y0_p)) \
		    * np.exp(-(((sig_x0**2)*((2*np.pi*u_p0)**2))/2) - (((sig_y0**2)*((2*np.pi*v_p0)**2))/2))
	model1 = (I1/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p1*x1_p+v_p1*y1_p)) \
		    * np.exp(-(((sig_x1**2)*((2*np.pi*u_p1)**2))/2) - (((sig_y1**2)*((2*np.pi*v_p1)**2))/2))	
	model = model0 + model1
	inv_sigma2 = weights 
	# phs_model = np.arctan(np.imag(model)/np.real(model))
	# phs_vis = np.arctan(np.imag(vis)/np.real(vis))
	return -0.5*(np.sum((np.real(vis) - np.real(model))**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_real2(pars,vis):
	x0, y0, x1, y1 = pars 
	
	sun_diam_rad = Angle(0.5*u.deg).rad

	if -2*sun_diam_rad < x0 < -0.25*sun_diam_rad and -2*sun_diam_rad < y0 < -0.25*sun_diam_rad \
	and -2*sun_diam_rad < x1 < -0.25*sun_diam_rad and -2*sun_diam_rad < y1 < -0.25*sun_diam_rad:
		return 0.0
	return -np.inf

def lnprob_real2(pars, u,v,vis,weights):
	lp = lnprior_real2(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real2(pars, u,v,vis,weights)

def fringe_pattern(u,v,x0,y0,theta):
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)
	fringe = np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p))	
	return fringe


def two_gauss_V(u,v, I0,x0,y0,sig_x0,sig_y0,theta0,I1,x1,y1,sig_x1,sig_y1,theta1):
	
	V0 = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0)
	V1 = gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1)

	V = V0 + V1
	return V

def two_gauss_I(x,y, I0,x0,y0,sig_x0,sig_y0,theta0,I1,x1,y1,sig_x1,sig_y1,theta1):
	
	I0 = gauss_I_theta(x,y,I0,x0,y0,sig_x0,sig_y0,theta0)
	I1 = gauss_I_theta(x,y,I1,x1,y1,sig_x1,sig_y1,theta1)

	I = I0 + I1
	return I

def gauss_I(x,y,I0,x0,y0,sig_x,sig_y):

	I = (I0/(2*np.pi*sig_x*sig_y)) * np.exp( -( (x-x0)**2/(2*(sig_x**2)) ) - ( (y-y0)**2/(2*(sig_y**2)) ) )

	return I

def gauss_I_theta(x,y,I0,x0,y0,sig_x,sig_y,theta):
	
	a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
	b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
	c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))

	I = (I0/(2*np.pi*sig_x*sig_y)) * np.exp(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

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

# data = data[0,:] + data[3,:]
# V = vis[0,:] + vis[3,:]

# vis_err = np.sqrt(1/weights*abs(vis))
# vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
# weights = (weights[0,:]*abs(vis)[0,:] + weights[3,:]*abs(vis)[3,:])

vis_file = "SB076MS_data.npz"  
sb = int(vis_file.split("SB")[-1][:3])
epoch_start = datetime(1858,11,17) #MJD

# load_vis = np.load(vis_file)
# freq = float(load_vis["freq"])
# delt = float(load_vis["dt"])
# delf = float(load_vis["df"])
# wlen = c.value/freq
# uvws = load_vis["uvws"]/wlen
# times = load_vis["times"]
# data = load_vis["data"]
# vis = load_vis["vis"]
# weights = load_vis["weights"]
# ant0 = load_vis["ant0"]
# ant1 = load_vis["ant1"]

# """
# eig is largest eigenvalue of gain matrix G == ||G|| 
# see https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/data-weights (accessed 16/09/2019)
# """

# auto_corrs = np.where(ant0==ant1)[0]
# cross_corrs = np.where(ant0!=ant1)[0]

# data = data[0,:] + data[3,:]
# V = vis[0,:] + vis[3,:]

# vis_err = np.sqrt(1/weights*abs(vis))
# vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
# weights = (weights[0,:]*abs(vis)[0,:] + weights[3,:]*abs(vis)[3,:])
# averaged weight is sum of weights according to 
# https://www.astron.nl/lofarwiki/lib/exe/fetch.php?media=public:user_software:documentation:ndppp_weights.pdf (accessed 16/09/2019)
# and maths, I guess

sun_diam_rad = Angle(0.5*u.deg).rad
sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

stria_oom = Angle(0.1*u.arcmin).rad
sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

scatter_diam = Angle(10*u.arcmin).rad
sig_scatter = scatter_diam/(2*np.sqrt(2*np.log(2)))

sig_x_guess = sig_sun
sig_y_guess = sig_sun


no_weight = np.ones(630)

vis = LOFAR_vis(vis_file, 1216)
q_sun = vis.queit_sun_df()
burst = vis.vis_df()

params = Parameters()
params.add_many(('I0',5*np.max(np.real(burst.bg_vis)),True,0,abs(np.max(burst.bg_vis))*10), 
	('x0',-sun_diam_rad,True,-2*sun_diam_rad,-0.5*sun_diam_rad),
	('y0',-0.5*sun_diam_rad,True,-1*sun_diam_rad,-0.5*sun_diam_rad), 
	('sig_x',sig_x_guess,True,sig_stria,2*sig_sun),
	('sig_y',sig_y_guess,True,sig_stria,2*sig_sun), 
	('theta',np.pi/4,True,0,np.pi))

# fit = minimize(residual, params, method="leastsq", args=(q_sun.u, q_sun.v, q_sun.bg_vis, q_sun.weight))
fit = minimize(residual_abs, params, method="leastsq", args=(burst.u, burst.v, burst.bg_vis, burst.weight))

fit.params["I0"].vary = False
fit.params["sig_x"].vary = False
fit.params["sig_y"].vary = False
fit.params["theta"].vary = False

fit_pos = minimize(residual_pos, fit.params, method="leastsq", args=(burst.u, burst.v, burst.bg_vis, burst.weight))
print(fit_pos.params.valuesdict())

val_dict = fit_pos.params.valuesdict()
g_fit = gauss_2D(burst.u, burst.v, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
	val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'])

plt.figure()
plt.plot(burst.ang_scales, abs(burst.bg_vis),'o')
plt.plot(burst.ang_scales, abs(g_fit),'r+') 
plt.xscale('log')

arr_size = 5000
u_arr = np.arange(q_sun.u.min(),q_sun.u.max(),(q_sun.u.max()-q_sun.u.min())/arr_size )
v_arr = np.arange(q_sun.v.min(),q_sun.v.max(),(q_sun.v.max()-q_sun.v.min())/arr_size )
uv_mesh = np.meshgrid(u_arr,v_arr) 
x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
xy_mesh = np.meshgrid(x_arr,y_arr) 

sun_pos = SkyCoord(vis.sun_dir[0].rad, vis.sun_dir[1].rad, unit="rad")   


I_fit = gauss_I_theta(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
	val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'])
# fig, ax = plt.subplots()
# ax.imshow(I_fit, aspect='equal', origin='lower', extent=[x_arr[0], x_arr[-1], y_arr[0], y_arr[-1]])
# s = Circle((vis.solar_ra_offset.rad,vis.solar_dec_offset.rad),vis.solar_rad.rad, color='r', fill=False)
# ax.add_patch(s)

gm_fit = gauss_2D(uv_mesh[0], uv_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
	val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'])
# fig, ax = plt.subplots()
# ax.imshow(gm_fit.real, aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
# s = Circle((0,0),1/vis.solar_rad.rad, color='r', fill=False)
# ax.add_patch(s)

# us = np.linspace(-10,10,500)
# vs = np.linspace(-10,10,500)
# uvmesh = np.meshgrid(us,vs)
# g2dm = gauss_2D(uvmesh[0], uvmesh[1], 10, 3e-1, 4e-1, 1e-1, 2e-1, np.pi/3)

# np.random.seed(42)
# us = np.random.choice(us, len(us))
# vs = np.random.choice(vs, len(vs))


# g2d = gauss_2D(us, vs, 10, 3e-1, 4e-1, 1e-1, 2e-1, np.pi/3)

# # plt.figure()
# # plt.scatter(us,vs,c=g2d.real)

# params = Parameters()
# params.add_many(('I0',5,True,1), 
# 	('x0',1e-1,True,-1,1),
# 	('y0',1e-1,True,-1,1), 
# 	('sig_x',1e-1,True,0,1),
# 	('sig_y',1e-1,True,0,1), 
# 	('theta',np.pi/3,True,0,np.pi/2))



# fit = minimize(residual, params, method="leastsq", args=(us, vs, g2d))
# print(fit.params.valuesdict())

# df_auto_list = []
# df_cross_list = []
# t_s = 48
# t_e = 49#len(V)

# for t in range(t_s,t_e):
# 	d_auto = {"u":uvws[0,t,:][auto_corrs],"v":uvws[1,t,:][auto_corrs],"w":uvws[2,t,:][auto_corrs], 
# 	"times":times[t,auto_corrs], "vis":V[t,auto_corrs], "vis_err":vis_err[t,auto_corrs],"weight":weights[t,auto_corrs]}
# 	d_cross = {"u":uvws[0,t,:][cross_corrs],"v":uvws[1,t,:][cross_corrs],"w":uvws[2,t,:][cross_corrs], 
# 	"times":times[t,cross_corrs], "vis":V[t,cross_corrs], "vis_err":vis_err[t,cross_corrs],"weight":weights[t,cross_corrs]}

# 	df_auto = pd.DataFrame(data=d_auto) 
# 	df_cross = pd.DataFrame(data=d_cross) 

# 	df_auto_list.append(df_auto)
# 	df_cross_list.append(df_cross)

# t = t_s
# pars_list = []
# for df_cross in df_cross_list:
# 	print("Fitting for t={}".format(str(t).zfill(3)))
# 	uv_dist = np.sqrt(df_cross.u**2 + df_cross.v**2)
# 	ang_scales = Angle((1/uv_dist)*u.rad)
# 	bg = np.where(ang_scales.arcmin < 2 )[0]
# 	bg_vis = df_cross.vis[bg]
# 	bg_mean = np.mean(bg_vis)
# 	bg_mean_abs = np.mean(abs(bg_vis))
# 	# vis_scale = (df_cross.vis - np.min(abs(df_cross.vis)))/(np.max(abs(df_cross.vis))-np.min(abs(df_cross.vis)))
# 	# weight_scale = (df_cross.weight - np.min(abs(df_cross.weight)))/(np.max(abs(df_cross.weight))-np.min(abs(df_cross.weight)))
# 	df_cross = df_cross.assign(bg_vis = (df_cross.vis - bg_mean))
# 	df_cross = df_cross.assign(abs_bg_vis = (abs(df_cross.vis) - bg_mean_abs))
# 	# df_cross = df_cross.assign(scale = vis_scale)
# 	# df_cross = df_cross.assign(weight_scale = weight_scale)

# 	sun_params = Parameters()
# 	sun_params.add_many(('I0',5*np.max(np.real(df_cross.vis)),True,0,abs(np.max(df_cross.vis))*10), 
# 		('x0',0,True,-0.1*sun_diam_rad,0.1*sun_diam_rad),
# 		('y0',0,True,-0.1*sun_diam_rad,0.1*sun_diam_rad), 
# 		('sig_x',sig_sun,True,sig_sun-(0.1*sig_sun),sig_sun+(0.1*sig_sun)),
# 		('sig_y',sig_sun,True,sig_sun-(0.1*sig_sun),sig_sun+(0.1*sig_sun)), 
# 		('theta',0,False))
# 	smodel = Model(gauss_2D, independent_vars=['u','v'])
# 	sresult = smodel.fit(df_cross.bg_vis, u=df_cross.u, v=df_cross.v, params=sun_params, weights=df_cross.weight)

# 	df_cross = df_cross.assign(nosun_vis = df_cross.bg_vis-sresult.best_fit)
# #params use starting values determined by playing around with it.

# 	# params = Parameters()
# 	# params.add_many(('I0',np.max(df_cross.abs_bg_vis)), 
# 	# 	('sig_x0',sig_x_guess),
# 	# 	('sig_y0',sig_y_guess), 
# 	# 	('theta0',np.pi/3),
# 	# 	('I1',np.max(df_cross.abs_bg_vis)), 
# 	# 	('sig_x1',0.5*sig_x_guess),
# 	# 	('sig_y1',0.5*sig_y_guess), 
# 	# 	('theta1',np.pi/3))


# 	# # gmodel = Model(gauss_2D, independent_vars=['u','v']) 
# 	# # result = gmodel.fit(df_cross.vis, u=df_cross.u, v=df_cross.v, params=params, weights=df_cross.weight)
# 	# # pars = result.params 

# 	# ndim, nwalkers = 8, 500
# 	# nsamples = 1000
# 	# guess = [*params.valuesdict().values()]
# 	# guess = np.array(guess)
# 	# pos = [guess + guess*1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# 	# with Pool() as ep:
# 	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_abs2, args=(df_cross.u,df_cross.v,df_cross.abs_bg_vis,df_cross.weight), pool=ep)
# 	# 	sampler.run_mcmc(pos,nsamples)
# 	# 	ep.close()
# 	# fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True) 
# 	# samples = sampler.chain 
# 	# labels = ["I0", "sig_x0","sig_y0","theta0","I1", "sig_x1","sig_y1","theta1"] 
# 	# for i in range(ndim): 
# 	# 	ax = axes[i] 
# 	# 	ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 	# 	ax.set_ylabel(labels[i]) 
# 	# 	ax.yaxis.set_label_coords(-0.1, 0.5) 
	 
# 	# axes[-1].set_xlabel("step number");
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_chain_abs_t{}_2gauss.png".format(str(t).zfill(3)))

# 	# samples = sampler.chain[:, 20:, :].reshape((-1, ndim))

# 	# mc_pars = np.percentile(samples,[16,50,84],axis=0)
# 	# c_plot = corner.corner(samples, labels=["I0", "sig_x0","sig_y0","theta0","I1", "sig_x1","sig_y1","theta1"], truths=[*mc_pars[1]]) 
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_corner_abs_t{}_2gauss.png".format(str(t).zfill(3)))

# 	# phs_ndim = 4
# 	# phs_params = Parameters()
# 	# phs_params.add_many(('x0',-sun_diam_rad),
# 	# 	('y0',-sun_diam_rad/2),
# 	# 	('x1',-sun_diam_rad),
# 	# 	('y1',-sun_diam_rad/2))
	

# 	# phs_guess = [*phs_params.valuesdict().values()]
# 	# phs_guess = np.array(phs_guess)
# 	# phs_pos = [phs_guess + phs_guess*1e-4*np.random.randn(phs_ndim) for i in range(nwalkers)]
# 	# with Pool() as ep:
# 	# 	phs_sampler = emcee.EnsembleSampler(nwalkers, phs_ndim, lnprob_real2, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
# 	# 	phs_sampler.run_mcmc(phs_pos,nsamples)
# 	# 	ep.close()

# 	# fig, axes = plt.subplots(phs_ndim, figsize=(10, 7), sharex=True) 
# 	# phs_samples = phs_sampler.chain 
# 	# labels = ["x0","y0","x1","y1"] 
# 	# for i in range(phs_ndim): 
# 	# 	ax = axes[i] 
# 	# 	ax.plot(phs_sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 	# 	ax.set_ylabel(labels[i]) 
# 	# 	ax.yaxis.set_label_coords(-0.1, 0.5) 
	 
# 	# axes[-1].set_xlabel("step number");
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_chain_xy_t{}_2gauss.png".format(str(t).zfill(3)))

# 	# phs_samples = phs_sampler.chain[:, 20:, :].reshape((-1, phs_ndim))

# 	# phs_mc_pars = np.percentile(phs_samples,[16,50,84],axis=0)
# 	# phs_c_plot = corner.corner(phs_samples, labels=["x0","y0","x1","y1"], truths=[*phs_mc_pars[1]]) 
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_corner_xy_t{}_2gauss.png".format(str(t).zfill(3)))


# 	# two_fit_pars = np.insert(mc_pars[1],1,phs_mc_pars[1])
# 	# # # pars_list.append(two_fit_pars)

# 	# # """
# 	# # Save every 10th run
# 	# # """
# 	# # if t % 10 == 0:
# 	# # 	print("Saving at t={}".format(str(t)))
# 	# # 	# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)

# 	# mcg = two_gauss_V(df_cross.u, df_cross.v, *two_fit_pars)

# 	# plt.figure()
# 	# plt.plot(ang_scales.arcminute, abs(df_cross.vis),"o", label="data")
# 	# plt.plot(ang_scales.arcminute, abs(mcg),"r+", label="fit")
# 	# plt.title("Visibility vs Angular Scale")
# 	# plt.xlabel("Angular scale (arcminute)")
# 	# plt.ylabel("Visibility (AU)")
# 	# plt.xscale("log")
# 	# plt.legend()
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_absfit_t{}_2gauss.png".format(str(t).zfill(3)))

# 	# arr_size = 5000
# 	# u_arr = np.arange(df_cross.u.min(),df_cross.u.max(),(df_cross.u.max()-df_cross.u.min())/arr_size )
# 	# v_arr = np.arange(df_cross.v.min(),df_cross.v.max(),(df_cross.v.max()-df_cross.v.min())/arr_size )
# 	# uv_mesh = np.meshgrid(u_arr,v_arr) 
# 	# x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
# 	# y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
# 	# xy_mesh = np.meshgrid(x_arr,y_arr) 

# 	# two_fit_I = gauss_I_theta(xy_mesh[0], xy_mesh[1], *two_fit_pars)


# 	# fig, ax = plt.subplots() 
# 	# ax.imshow(two_fit_I, origin='lower',extent=[Angle(x_arr[0]*u.rad).arcsec, Angle(x_arr[-1]*u.rad).arcsec,
# 	# 	Angle(y_arr[0]*u.rad).arcsec, Angle(y_arr[-1]*u.rad).arcsec])
# 	# limb = Circle((0,0),Angle(15*u.arcmin).arcsec, color='r', fill=False)
# 	# ax.add_patch(limb)
# 	# plt.xlabel("X (arcsec)")
# 	# plt.ylabel("Y (arcsec)")
# 	# plt.title("mcmc_recreate")
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_recreate_t{}_2gauss.png".format(str(t).zfill(3)))
# 	# t+=1
# 	params = Parameters()
# 	params.add_many(('I0',5*np.max(np.real(df_cross.vis)),True,0,abs(np.max(df_cross.vis))*10), 
# 		('x0',-sun_diam_rad,True,-2*sun_diam_rad,2*sun_diam_rad),
# 		('y0',-sun_diam_rad,True,-2*sun_diam_rad,2*sun_diam_rad), 
# 		('sig_x',sig_x_guess,True,sig_stria,2*sig_sun),
# 		('sig_y',sig_y_guess,True,sig_stria,2*sig_sun), 
# 		('theta',np.pi/3,True,-np.pi/2,np.pi/2))


# 	gmodel = Model(gauss_2D, independent_vars=['u','v']) 
# 	result = gmodel.fit(df_cross.vis, u=df_cross.u, v=df_cross.v, params=params, weights=df_cross.weight)
# 	pars = result.params 

# 	ndim, nwalkers = 4, 500
# 	nsamples = 150
# 	guess = [*params.valuesdict().values()]
# 	guess.pop(1)
# 	guess.pop(1)
# 	guess = np.array(guess)
# 	pos = [guess + guess*1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# 	with Pool() as ep:
# 		sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_abs, args=(df_cross.u,df_cross.v,df_cross.abs_bg_vis,df_cross.weight), pool=ep)
# 		sampler.run_mcmc(pos,nsamples)
# 		ep.close()

# 	fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True) 
# 	samples = sampler.chain 
# 	labels = ["I0", "sig_x","sig_y","theta"] 
# 	for i in range(ndim): 
# 		ax = axes[i] 
# 		ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 		ax.set_ylabel(labels[i]) 
# 		ax.yaxis.set_label_coords(-0.1, 0.5) 
	 
# 	axes[-1].set_xlabel("step number");
# 	plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_chain_abs_t{}.png".format(str(t).zfill(3)))

# 	samples = sampler.chain[:, 20:, :].reshape((-1, ndim))

# 	mc_pars = np.percentile(samples,[16,50,84],axis=0)
# 	c_plot = corner.corner(samples, labels=["I0", "sig_x","sig_y", "theta"], truths=[*mc_pars[1]]) 
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_corner_abs_t{}.png".format(str(t).zfill(3)))

# 	real_ndim = 2
# 	real_params = Parameters()
# 	real_params.add_many(('x0',-sun_diam_rad),
# 		('y0',-sun_diam_rad/2))
	
# 	real_guess = [*real_params.valuesdict().values()]
# 	real_guess = np.array(real_guess)
# 	real_pos = [real_guess + real_guess*1e-4*np.random.randn(real_ndim) for i in range(nwalkers)]
# 	with Pool() as ep:
# 		real_sampler = emcee.EnsembleSampler(nwalkers, real_ndim, lnprob_real, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
# 		real_sampler.run_mcmc(real_pos,nsamples)
# 		ep.close()

# 	fig, axes = plt.subplots(real_ndim, figsize=(10, 7), sharex=True) 
# 	real_samples = real_sampler.chain 
# 	labels = ["x0","y0"] 
# 	for i in range(real_ndim): 
# 		ax = axes[i] 
# 		ax.plot(real_sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 		ax.set_ylabel(labels[i]) 
# 		ax.yaxis.set_label_coords(-0.1, 0.5) 
	 
# 	axes[-1].set_xlabel("step number");
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_chain_xy_t{}.png".format(str(t).zfill(3)))

# 	real_samples = real_sampler.chain[:, 20:, :].reshape((-1, real_ndim))

# 	real_mc_pars = np.percentile(real_samples,[16,50,84],axis=0)
# 	real_c_plot = corner.corner(real_samples, labels=["x0","y0"], truths=[*real_mc_pars[1]]) 
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_corner_xy_t{}.png".format(str(t).zfill(3)))


# 	two_fit_pars = np.insert(mc_pars[1],1,real_mc_pars[1])
# 	pars_list.append(two_fit_pars)

# 	"""
# 	Save every 10th run
# 	"""
# 	if t % 10 == 0:
# 		print("Saving at t={}".format(str(t)))
# 		# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)

# 	mcg = gauss_2D(df_cross.u, df_cross.v, *two_fit_pars)

# 	plt.figure()
# 	plt.plot(ang_scales.arcminute, abs(df_cross.abs_bg_vis),"o", label="data")
# 	plt.plot(ang_scales.arcminute, abs(mcg),"r+", label="fit")
# 	plt.title("Visibility vs Angular Scale")
# 	plt.xlabel("Angular scale (arcminute)")
# 	plt.ylabel("Visibility (AU)")
# 	plt.xscale("log")
# 	plt.legend()
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_absfit_t{}.png".format(str(t).zfill(3)))

# 	arr_size = 5000
# 	u_arr = np.arange(df_cross.u.min(),df_cross.u.max(),(df_cross.u.max()-df_cross.u.min())/arr_size )
# 	v_arr = np.arange(df_cross.v.min(),df_cross.v.max(),(df_cross.v.max()-df_cross.v.min())/arr_size )
# 	uv_mesh = np.meshgrid(u_arr,v_arr) 
# 	x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
# 	y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
# 	xy_mesh = np.meshgrid(x_arr,y_arr) 

# 	two_fit_I = gauss_I_theta(xy_mesh[0], xy_mesh[1], *two_fit_pars)


# 	fig, ax = plt.subplots() 
# 	ax.imshow(two_fit_I, origin='lower',extent=[Angle(x_arr[0]*u.rad).arcsec, Angle(x_arr[-1]*u.rad).arcsec,
# 		Angle(y_arr[0]*u.rad).arcsec, Angle(y_arr[-1]*u.rad).arcsec])
# 	limb = Circle((0,0),Angle(15*u.arcmin).arcsec, color='r', fill=False)
# 	ax.add_patch(limb)
# 	plt.xlabel("X (arcsec)")
# 	plt.ylabel("Y (arcsec)")
# 	plt.title("mcmc_recreate")
# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/mcmc/SB076/mcmc_recreate_t{}.png".format(str(t).zfill(3)))
# 	t+=1

# 	# plt.close('all')
# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)
t_run = time.time()-t0
print("Time to run:", t_run)
plt.show()

