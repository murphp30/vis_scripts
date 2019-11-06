#!/usr/bin/env python

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.patches import Circle
from datetime import datetime, timedelta
from scipy import interpolate
import scipy.optimize as opt
from astropy.constants import c, m_e, R_sun,e, eps0
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

def dirac_delta(u,v):
	if u and v == 0:
		return np.inf
	else:
		return 0

def Newkirk(r):
    n_0 = 4.2e4
    n = n_0*10**(4.32*R_sun.value/r) #in cm^-3
    return n*1e6

def dist_from_freq(freq):
	kappa = np.sqrt((e.value**2/(m_e.value*eps0.value)))/(2*np.pi)
	n_0 = 4.2e4 * 1e6 #cm^-3 to m^-3, keeping it SI
	r = R_sun.value * (2.16)/(np.log10(freq)-np.log10(kappa*np.sqrt(n_0)))
	return r

def ang_scatter(r,freq,e_sq_over_h):
	#e_sq_over_h = 5e-8 # m^-1 Steinberg et. al. 1971, Riddle 1974, Chrysaphi et. al. 2018
	f_p =np.sqrt((e.value**2*Newkirk(r))/(eps0.value*m_e.value))/(2*np.pi)
	return (np.sqrt(np.pi)/2) * ((f_p**4)/((freq**2-f_p**2)**2) )* e_sq_over_h

def rad_to_m(rad):
	R_rad = Angle(15*u.arcmin).rad
	return rad*(R_sun.value/R_rad)

def ellipse(x,y,I0,x0,y0,bmin,bmaj,theta,C):
	x_p = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)
	y_p = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
	el = x_p**2/bmaj**2 + y_p**2/bmin**2
	el_ar = np.full((len(x), len(y)),C)
	el_ar[np.where(el<1)] = I0	
	return el_ar

def gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	u_p,v_p  =  rotate_coords(u,v,theta)#u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	#x0_p, y0_p = rotate_coords(x0,y0,theta)

	V =  np.exp(-2*np.pi*1j*(u*x0+v*y0)) \
	* ((I0/(2*np.pi)) * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) + C)
	
	return V

def ln_gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	u_p,v_p  =  rotate_coords(u,v,theta)#u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)
	#x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = np.log(I0/(2*np.pi)) + (-2*np.pi*1j*(u*x0+v*y0)) \
	+ (-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))
	
	return V

def residual(pars, u,v, data, fit_model="gauss", weights=None, ngauss=1, size=True):
	parvals = pars.valuesdict()
	
	if ngauss == 1:
		I0 = parvals['I0']
		sig_x = parvals['sig_x']
		sig_y = parvals['sig_y']
		theta = parvals['theta']
		C = parvals['C']
		if size:
			x0 = 0
			y0 = 0
		
		else:
			x0 = parvals['x0']
			y0 = parvals['y0']
		if fit_model == "gauss":
			model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C)
		elif fit_model == "ellipse": #this doesn't actually work and I probably don't need it.
			model = np.fft.fftshift(np.fft.fft2(ellipse(1/u,1/v,I0,x0,y0,sig_x,sig_y, theta,C)))
	elif ngauss == 2:
		I0 = parvals['I0']
		sig_x0 = parvals['sig_x0']
		sig_y0 = parvals['sig_y0']
		theta0 = parvals['theta0']
		C0 = parvals['C0']
		I1 = parvals['I1']
		sig_x1 = parvals['sig_x1']
		sig_y1 = parvals['sig_y1']
		theta1 = parvals['theta1']
		#C1 = parvals['C1']
		if size: #assume (x0,y0) and (x1,y1) are close
			x0 = 0
			y0 = 0
			x1 = 0
			y1 = 0		
		else:
			x0 = parvals['x0']
			y0 = parvals['y0']
			x1 = parvals['x1']
			y1 = parvals['y1']
		if fit_model == "gauss":
			model = two_gauss_V(u,v,I0,x0,y0,sig_x0,sig_y0,theta0,C0,\
			I1,x1,y1,sig_x1,sig_y1,theta1)
			m0,m1 = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0,C0),  gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1,C0)
	else:
		print("Must have max 2 gauss (for now)")
		return
	
	if size:
		# if ngauss == 2:
		# 	if weights is None:
		# 		#abs then log otherwise you get fringes in recreated image
		# 		resid = np.log(m0+m1) - np.log(abs(data)) #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
		# 	else:
		# 		resid = (np.log(m0+m1) - np.log(abs(data)))*weights#np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
		# elif ngauss == 1:
		if weights is None:
			#abs then log otherwise you get fringes in recreated image
			resid = np.log(abs(model)) - np.log(abs(data)) #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
		else:
			resid = (np.log(abs(model)) - np.log(abs(data)))*weights#np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
	
	else:
		if weights is None:
			resid = np.log((model.real - data.real)**2) + np.log((model.imag-data.imag)**2) #np.angle(model) - np.angle(data)
		else:
			resid = (np.log((model.real - data.real)**2) + np.log((model.imag-data.imag)**2))*weights#(np.angle(model)-np.angle(data))*weights		
	return resid

def lnlike(pars, u,v,vis,weights=None, size=True):
	try:
		x0,y0 = pars
	except ValueError:
		I0,sig_x,sig_y,theta, C = pars
	
	if size:
		x0 = 0
		y0 = 0
	else:
		size_fit = [*fit.params.valuesdict().values()]
		I0, sig_x, sig_y, theta, C = size_fit[0], *size_fit[3:]
	u_p,v_p  =  rotate_coords(u,v,theta)
	model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C)
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
		I0,sig_x,sig_y,theta,C = pars
	if size:
		x0 = 0
		y0 = 0
	else:
		size_fit = [*fit.params.valuesdict().values()]
		I0, sig_x, sig_y, C = size_fit[0], *size_fit[3:-1]
	
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

def two_gauss_V(u,v, I0,x0,y0,sig_x0,sig_y0,theta0,C0,I1,x1,y1,sig_x1,sig_y1,theta1):
	
	V0 = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0,C0)
	V1 = gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1,C0)

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
		

		uvws = self.load_vis["uvws"]/self.wlen
		times = self.load_vis["times"]
		times = epoch_start + timedelta(seconds=1)*times
		data = self.load_vis["data"]
		vis = self.load_vis["vis"]
		weights = self.load_vis["weights"]
		flag = self.load_vis["flag"]
		data = data[0,:] + data[3,:]
		V = vis[0,:] + vis[3,:]
		flag = flag[0,:] + flag[3,:]
		vis_err = np.sqrt(1/weights*abs(vis))
		vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
		weights = weights[0,:] + weights[3,:]#(weights[0,:]*abs(vis)[0,:] + weights[3,:]*abs(vis)[3,:])		
		#weights[flag] = 0
		flag = np.invert(flag)
		ntimes =flag.shape[0]
		uvws=uvws[:,flag].reshape(3,ntimes,-1)
		times = times[flag].reshape(ntimes,-1)
		data = data[flag].reshape(ntimes,-1)
		V = V[flag].reshape(ntimes,-1)
		vis_err = vis_err[flag].reshape(ntimes,-1)
		weights = weights[flag].reshape(ntimes,-1)

		ant0 = ant0[flag[0]] #assume flag only in baseline not time
		ant1 = ant1[flag[0]]
		cross_corrs = np.where(ant0!=ant1)[0]
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
		flag = self.load_vis["flag"][:,:q_t,:]
		data = data[0,:] + data[3,:]
		flag = flag[0,:] + flag[3,:]
		V = vis[0,:] + vis[3,:]
		vis_err = np.sqrt(1/weights*abs(vis))
		vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
		weights = (weights[0,:] + weights[3,:])		
		
		flag = np.invert(flag)
		ntimes =flag.shape[0]
		uvws=uvws[:,flag].reshape(3,ntimes,-1)
		
		data = data[flag].reshape(ntimes,-1)
		V = V[flag].reshape(ntimes,-1)
		vis_err = vis_err[flag].reshape(ntimes,-1)
		weights = weights[flag].reshape(ntimes,-1)

		ant0 = ant0[flag[0]] #assume flag only in baseline not time
		ant1 = ant1[flag[0]]
		cross_corrs = np.where(ant0!=ant1)[0]
		uvws = np.mean(uvws, axis=1)
		V = np.mean(V, axis=0)
		data = np.mean(data, axis=0)
		vis_err = np.mean(vis_err, axis=0)
		weights = np.sum(weights, axis=0)


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

ang_arr = np.arange(0, 600, 600/arr_size)

bf_data, bf_freq, bf_tarr = get_data(bf_file, vis0.obsstart, vis0.obsend )
bf_delt = bf_tarr[1] - bf_tarr[0]
bf_delf = bf_freq[1] - bf_freq[0]

burst_delt = int(np.round((vis0.delt)/bf_delt))
burst_f_mid_bf = np.where(bf_freq == vis0.freq*1e-6 +(bf_delf/2))[0][0]
burst_f_start_bf = burst_f_mid_bf - 8
burst_f_end_bf = burst_f_mid_bf + 8

bf_data_t = np.mean(bf_data[burst_delt*q_t:,burst_f_start_bf:burst_f_end_bf],axis=1)
day_start = datetime(2015,10,17,8,00,00)
bf_dt_arr = day_start + timedelta(seconds=1)*bf_tarr[burst_delt*q_t:]
bf_dt_arr = dates.date2num(bf_dt_arr)
date_format = dates.DateFormatter("%H:%M:%S")

burst_max_t = burst_delt*q_t+np.argmax(bf_data_t) 

def parallel_fit(i):
	save = False
	vis = LOFAR_vis(vis_file, i)
	burst = vis.vis_df()

	ngauss = 1

	params = Parameters()
	fit_vis = burst.vis - q_sun.vis
	#fit_vis = np.log(fit_vis)
	fit_weight = np.sqrt(burst.weight**2 + q_sun.weight**2)
	# fit_weight = np.log(fit_weight)
	if ngauss == 2:
		params.add_many(('I0',np.pi*np.max(abs(fit_vis)),True,0,abs(np.max(fit_vis))*10), 
			('x0',-0.7*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad),
			('y0',-0.5*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x0',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y0',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta0',np.pi/3,True,0,np.pi),
			('C0',np.min(abs(fit_vis)),True, 0),
			('I1',np.pi*np.max(abs(fit_vis))/2,True,0,abs(np.max(fit_vis))*10), 
			('x1',-0.7*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad),
			('y1',-0.5*sun_diam_rad,False,-1.5*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x1',sig_x1_guess,True,sig_stria,1.5*sig_sun),
			('sig_y1',sig_y1_guess,True,sig_stria,1.5*sig_sun), 
			('theta1',np.pi/3,True,0,np.pi))
			#('C1',np.min(abs(fit_vis)),True, 0))
		# fit = minimize(residual, params, method="leastsq", args=(burst.u, burst.v, fit_vis,"gauss", fit_weight , ngauss, False))
	elif ngauss == 1:
		params.add_many(('I0',2*np.pi*np.max(abs(fit_vis)),True,0,abs(np.max(fit_vis))*10), 
			('x0',-0.7*sun_diam_rad,False,-2*sun_diam_rad,-0.25*sun_diam_rad),
			('y0',-0.5*sun_diam_rad,False,-2*sun_diam_rad,-0.25*sun_diam_rad), 
			('sig_x',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta',np.pi/3,True,0, np.pi),
			('C',np.mean(abs(fit_vis)),True, np.min(abs(fit_vis))))


	fit = minimize(residual, params, method="leastsq", args=(burst.u, burst.v, fit_vis,"gauss", fit_weight , ngauss, True))
	print("Fitting", i-q_t)

	if ngauss == 2:
		fit.params["I0"].vary = False
		fit.params["x0"].vary = True
		fit.params["y0"].vary = True
		fit.params["sig_x0"].vary = False
		fit.params["sig_y0"].vary = False
		fit.params["theta0"].vary = False
		fit.params["C0"].vary = False
		fit.params["I1"].vary = False
		fit.params["x1"].vary = True
		fit.params["y1"].vary = True
		fit.params["sig_x1"].vary = False
		fit.params["sig_y1"].vary = False
		fit.params["theta1"].vary = False
		# fit.params["C1"].vary = False
	elif ngauss == 1:
		fit.params["I0"].vary = False
		fit.params["x0"].vary = True
		fit.params["y0"].vary = True
		fit.params["sig_x"].vary = False
		fit.params["sig_y"].vary = False
		fit.params["theta"].vary = False
		fit.params["C"].vary = False

	fit = minimize(residual, fit.params, method="leastsq", args=(burst.u, burst.v, fit_vis,"gauss",fit_weight ,ngauss,False))
	val_dict = fit.params.valuesdict()

	if ngauss == 2:
		# g_fit = two_gauss_V(burst.u, burst.v, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
		# 	val_dict['sig_x0'], val_dict['sig_y0'], val_dict['theta0'], val_dict['C0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
		# 	val_dict['sig_x1'], val_dict['sig_y1'], val_dict['theta1'])
		u_rot0, v_rot0 = rotate_coords(u_arr, v_arr, val_dict['theta0'])
		u_rot1, v_rot1 = rotate_coords(u_arr, v_arr, val_dict['theta1'])
		g_fitx = ((val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_x0']**2 * (2*np.pi*u_rot0)**2))/2)) \
		+ ((val_dict['I1']/(2*np.pi)) * np.exp(-((val_dict['sig_x1']**2 * (2*np.pi*u_rot1)**2))/2)) + 2*val_dict['C0']
		g_fity = ((val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_y0']**2 * (2*np.pi*v_rot0)**2))/2)) \
		+ ((val_dict['I1']/(2*np.pi)) * np.exp(-((val_dict['sig_y1']**2 * (2*np.pi*v_rot1)**2))/2)) + 2*val_dict['C0']

		ang_u, ang_v = Angle((1/abs(u_rot0))*u.rad).arcmin, Angle((1/abs(v_rot0))*u.rad).arcmin
		ang_u1, ang_v1 = Angle((1/abs(u_rot1))*u.rad).arcmin, Angle((1/abs(v_rot1))*u.rad).arcmin
		cont_fit = two_gauss_V(uv_mesh[0], uv_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x0'], val_dict['sig_y0'], val_dict['theta0'], val_dict['C0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
			val_dict['sig_x1'], val_dict['sig_y1'], val_dict['theta1'])

		I_fit = two_gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x0'], val_dict['sig_y0'], -val_dict['theta0'], val_dict['I1'], val_dict['x1'], val_dict['y1'], 
			val_dict['sig_x1'], val_dict['sig_y1'], -val_dict['theta1'])

	elif ngauss == 1:
		
		# g_fit = gauss_2D(burst.u, burst.v, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
					# val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'], val_dict['C'])
		u_rot, v_rot = rotate_coords(u_arr, v_arr, val_dict['theta'])
		g_fitx = (val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_x']**2 * (2*np.pi*u_rot)**2))/2) + val_dict['C']
		g_fity = (val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_y']**2 * (2*np.pi*v_rot)**2))/2) + val_dict['C']
		ang_u, ang_v = Angle((1/abs(u_rot))*u.rad).arcmin, Angle((1/abs(v_rot))*u.rad).arcmin
		cont_fit = gauss_2D(uv_mesh[0], uv_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'],
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'], val_dict['C'])
		I_fit = gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], -val_dict['theta']) 
		

		"""
		intensity is rotated opposite to visibility space (should fix this in function definition)
		"""
	# plt.figure()
	fig, ax = plt.subplots()#1,1, figsize=(8,7))
	ax.plot(burst.ang_scales, (abs(fit_vis)),'o')
	#ax.plot(burst.ang_scales, (abs(g_fit)),'r+') 
	# if ngauss == 1:
	ax.plot(ang_u, g_fitx, 'r')
	ax.plot(ang_v, g_fity, 'r')
	# else:
		# ax.plot(burst.ang_scales, (abs(g_fit)),'r+')
	ax.set_xlabel("Angular Scale (arcminute)")
	ax.set_ylabel("Visibility (AU)")
	ax.set_title("Vis vs ang scale {}".format(vis.time.isoformat()))

	ax.set_xscale('log')
	ax.set_xlim([ax.get_xlim()[0], 1e3])
	ax.set_ylim([ax.get_ylim()[0], 0.9e7])
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/vis_ang_scale_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/vis_ang_scale_raw.png")
		plt.close()	
	# plt.figure()
	# plt.plot(burst.ang_scales, np.log(abs(fit_vis))-np.log(abs(g_fit)),'o') 
	# plt.xlabel("Angular Scale (arcminute)")
	# plt.ylabel("Residual Visibility (AU)")
	# plt.title("Residual vs angular scale {}".format(vis.time.isoformat()))
	# plt.xscale('log')
	# bf_data_f = bf_data[burst_delt*(i+q_t),:]
	# fig, ax1 = plt.subplots()
	# ax1.plot(bf_dt_arr,bf_data_t)
	# ax1.xaxis_date()
	# ax1.set_ylim(0,None)
	# ax1.vlines(bf_dt_arr[burst_delt*(i-q_t)], ymin=0, ymax=bf_data_t[burst_delt*(i-q_t)],color='r')
	# ax1.set_ylabel("Intensity (AU)")
	# ax1.set_title("Intensity at {} MHz".format(np.round(vis.freq*1e-6,3)))
	# ax1.set_xlabel("Time")
	# ax1.xaxis.set_major_formatter(date_format)
	# plt.tight_layout()


	# plt.figure()
	# plt.scatter(burst.u, burst.v, c=np.log(abs(fit_vis)))
	# plt.xlabel(r"u ($\lambda$)")
	# plt.ylabel(r"v ($\lambda$)")
	# plt.title("uv plane")
	# if save:
	# 	plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/uv_plane_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
	# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/uv_raw.png")
	# 	plt.close()

	plt.figure()
	plt.scatter(burst.u, burst.v, c=np.log(abs(fit_vis)))
	# plt.imshow(np.log(abs(cont_fit)), aspect='auto', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]],
	#  vmin=np.min(np.log(abs(fit_vis))), vmax=np.max(np.log(abs(fit_vis))))
	plt.contour(uv_mesh[0], uv_mesh[1], np.log(abs(cont_fit)), 
		[np.log(0.1) + np.max(np.log(abs(cont_fit))),np.log(0.5) + np.max(np.log(abs(cont_fit))),
		np.log(0.9) + np.max(np.log(abs(cont_fit))),np.log(0.95) + np.max(np.log(abs(cont_fit)))],
		colors='r')
	# plt.contour(uv_mesh[0], uv_mesh[1], abs(np.log(cont_fit)), 
	# 	[0.9*np.max(abs(np.log(cont_fit)))],
	# 	colors='b')

	plt.xlim(-1000,1000)
	plt.ylim(-1000,1000)
	plt.xlabel(r"u ($\lambda$)")
	plt.ylabel(r"v ($\lambda$)")
	plt.title("uv plane")
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/uv_zoom_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/uv_cont_raw.png")
		plt.close()
	# plt.figure()
	# plt.imshow(np.log(abs(cont_fit)), aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
	# plt.xlim(-1000,1000)
	# plt.ylim(-1000,1000)
	# # plt.colorbar()
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
		plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/im_recreate_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/lmfit/tests/im_recreate_raw.png")
		plt.close()
	return fit
	#fig, ax = plt.subplots()
	#ax.imshow(abs(gm_fit), aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
	# s = Circle((0,0),1/vis.solar_rad.rad, color='r', fill=False)
	# ax.add_patch(s)

	
# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)
# with Pool() as p_fit:
# 	fits = p_fit.map(parallel_fit, range(q_t, q_t+79))
fit_pos = parallel_fit(q_t+24)
t_run = time.time()-t0
print("Time to run:", t_run)

plt.show()

