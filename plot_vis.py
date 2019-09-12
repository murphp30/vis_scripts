#!/usr/bin/env python

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime
from scipy import interpolate
import scipy.optimize as opt
from astropy.constants import c
from astropy.coordinates import Angle
import astropy.units as u
from lmfit import Model, Parameters, Minimizer,minimize
import corner
import emcee
from multiprocessing import Pool
import time

t0 = time.time()
def phase(Z):
	theta = np.arctan(Z.imag/Z.real)
	theta[np.where(Z.real == 0)] = 0
	return theta

def gauss_1D(x,A,mu,sig):
    return A*np.exp(-((x-mu)**2)/(2*(sig**2)))# + C

def left_gauss(x,A,mu,sig,C):
	g = np.zeros(len(x))
	for i in range(len(x)):
		if x[i] < mu:
			g[i] = A*np.exp(-((x[i]-mu)**2)/(2*(sig**2))) + C
		else:
			g[i] = A + C
	return g

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
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
	* np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) #+ C
	#(I0/(2*np.pi)) * np.exp(-(sig_x**2*(u_p-x0_p)**2/2)-(sig_y**2*(v_p-y0_p)**2/2)) #(I0/(2*np.pi)) #* np.exp(-1j*u_p*x0_p) * np.exp(-1j*v_p*y0_p)

	return V#.ravel()

def abs_gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
	* np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))# + C
	#(I0/(2*np.pi)) * np.exp(-(sig_x**2*(u_p-x0_p)**2/2)-(sig_y**2*(v_p-y0_p)**2/2)) #(I0/(2*np.pi)) #* np.exp(-1j*u_p*x0_p) * np.exp(-1j*v_p*y0_p)

	return abs(V)#.ravel()

def lnlike(pars, u,v,vis,weights):
	I0,x0,y0,sig_x,sig_y,theta = pars #,C
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	model = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
		    * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) #+ C)
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior(pars,vis):
	I0,x0,y0,sig_x,sig_y,theta = pars #,C 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
	sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

	if np.min(vis) < I0 < 10*np.max(vis) and -2*sun_diam_rad < x0 < 2*sun_diam_rad and \
 	-2*sun_diam_rad < y0 < 2*sun_diam_rad and sig_stria < sig_x < 2*sig_sun and sig_stria < sig_y < 2*sig_sun \
	and -np.pi/2 < theta < np.pi/2: #and 0 < C < 100*np.min(abs(vis)):
		return 0.0
	return -np.inf

def lnprob(pars, u,v,vis,weights):
	lp = lnprior(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(pars, u,v,vis,weights)

def lnlike_real(pars, u,v,vis,weights):
	x0, y0 = pars #,C
	I0, sig_x, sig_y, theta = [*mc_pars[1]]	
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)
	model = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
		    * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) #+ C)
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis.real-model.real)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_real(pars,vis):
	x0, y0 = pars #,C 
	I0, sig_x, sig_y, theta = [*mc_pars[1]]	
	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
	sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

	if -2*sun_diam_rad < x0 < 2*sun_diam_rad and \
 	-2*sun_diam_rad < y0 < 2*sun_diam_rad: #and 0 < C < 100*np.min(abs(vis)):
		return 0.0
	return -np.inf

def lnprob_real(pars, u,v,vis,weights):
	lp = lnprior_real(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_real(pars, u,v,vis,weights)

def lnlike_imag(pars, u,v,vis,weights):
	I0,x0,y0,sig_x,sig_y,theta = pars #,C
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	model = (I0/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) \
		    * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) #+ C)
	inv_sigma2 = weights 

	return -0.5*(np.sum((vis.imag-model.imag)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_imag(pars,vis):
	I0,x0,y0,sig_x,sig_y,theta = pars #,C 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
	sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

	if np.min(vis.imag) < I0 < 10*np.max(vis.imag) and -2*sun_diam_rad < x0 < 2*sun_diam_rad and \
 	-2*sun_diam_rad < y0 < 2*sun_diam_rad and sig_stria < sig_x < 2*sig_sun and sig_stria < sig_y < 2*sig_sun \
	and -np.pi/2 < theta < np.pi/2: #and 0 < C < 100*np.min(abs(vis)):
		return 0.0
	return -np.inf

def lnprob_imag(pars, u,v,vis,weights):
	lp = lnprior_imag(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_imag(pars, u,v,vis,weights)

def lnlike_abs(pars, u,v,vis,weights):
	I0,sig_x,sig_y,theta = pars #,C
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	model = abs((I0/(2*np.pi)) * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))) #+ C)
	inv_sigma2 = weights 

	return -0.5*(np.sum((abs(vis)-model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_abs(pars,vis):
	I0,sig_x,sig_y,theta = pars #,C 

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
	sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

	if np.min(abs(vis)) < I0 < 10*np.max(abs(vis)) and sig_stria < sig_x < 2*sig_sun and sig_stria < sig_y < 2*sig_sun \
	and 0 < theta < np.pi/2: #and 0 < C < 100*np.min(abs(vis)):
		return 0.0
	return -np.inf

def lnprob_abs(pars, u,v,vis,weights):
	lp = lnprior_abs(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_abs(pars, u,v,vis,weights)


def lnlike_phs(pars, u,v,vis,weights):
	x0,y0 = pars #,C
	theta = mc_pars[1,3]
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)
	model =  np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) #* np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2)) #+ C
	inv_sigma2 = weights 
	phs_model = np.arctan(model.imag/model.real)
	phs_vis = np.arctan(vis.imag/vis.real)

	return -0.5*(np.sum((phs_vis-phs_model)**2*inv_sigma2 - np.log(inv_sigma2)))

def lnprior_phs(pars,vis):
	x0,y0 = pars #,C
	theta = mc_pars[1,3]
	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(0.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
	sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

	if -4*sun_diam_rad < x0 < 4*sun_diam_rad and \
	-4*sun_diam_rad < y0 < 4*sun_diam_rad:# and mc_pars[0,3] < theta < mc_pars[2,3]: #and 0 < C < 100*np.min(abs(vis)):
		return 0.0
	return -np.inf

def lnprob_phs(pars, u,v,vis,weights):
	lp = lnprior_phs(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike_phs(pars, u,v,vis,weights)

def fringe_pattern(u,v,x0,y0,theta):
	u_p,v_p  =  u*np.cos(theta)+v*np.sin(theta),-u*np.sin(theta)+v*np.cos(theta)#rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)
	fringe = np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p))	
	return np.arctan(fringe.imag/fringe.real)

def res_gauss_2D(p,u,v,vis,weights):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	vals = p.valuesdict()
	u_p,v_p  =  u*np.cos(vals["theta"])+v*np.sin(vals["theta"]),-u*np.sin(vals["theta"])+v*np.cos(vals["theta"])#rotate_coords(u,v,vals["theta"])
	x0_p, y0_p = rotate_coords(vals["x0"],vals["y0"],vals["theta"])

	V = (((vals["I0"]/(2*np.pi)) * np.exp(-2*np.pi*1j*(u_p*x0_p+v_p*y0_p)) * \
	np.exp(-(((vals["sig_x"]**2)*((2*np.pi*u_p)**2))/2) - (((vals["sig_y"]**2)*((2*np.pi*v_p)**2))/2)) + vals["C"]) \
	 - vis)*weights
	#(I0/(2*np.pi)) * np.exp(-(sig_x**2*(u_p-x0_p)**2/2)-(sig_y**2*(v_p-y0_p)**2/2)) #(I0/(2*np.pi)) #* np.exp(-1j*u_p*x0_p) * np.exp(-1j*v_p*y0_p)

	return V.view(np.float128)
def log_gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta):
	"""
	gaussian is rotated, change coordinates to 
	find this angle
	"""
	u_p,v_p  =  rotate_coords(u,v,theta)
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	V = np.log((I0/(2*np.pi))) + ( -2*np.pi*1j*(u_p*x0_p+v_p*y0_p) ) + ( -( ( (sig_x**2)*((2*np.pi*u_p)**2))/2 ) - ( ((sig_y**2)*((2*np.pi*v_p)**2))/2 ) )
	

	return V#.ravel()

def two_gauss_V(u,v, I0,x0,y0,sig_x0,sig_y0,theta0,C0,I1,x1,y1,sig_x1,sig_y1,theta1,C1):
	
	V0 = gauss_2D(u,v,I0,x0,y0,sig_x0,sig_y0,theta0,C0)
	V1 = gauss_2D(u,v,I1,x1,y1,sig_x1,sig_y1,theta1,C1)

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

def gauss_2D_simple(u,v,I0,x0,y0,sig_x,sig_y):
	#u, v = uv

	V = (I0/(2*np.pi)) * np.exp( -2*np.pi*1j*(u*x0+v*y0) ) * np.exp( -( ( (sig_x**2)*((2*np.pi*u)**2))/2 ) - ( ((sig_y**2)*((2*np.pi*v)**2))/2 ) )
	return V

def bl_to_scale(bl):
	#give baseline in metres
	wvlen = c.value/freq #freq is a global variable taken from fits file
	return Angle(wvlen/bl,"rad").to(u.arcmin)

def scale_to_bl(scale):
	wvlen = c.value/freq
	return wlen/scale

def MC_fit(func, data, u, v, weight, niters=3):
	nparams = 7
	arr_len = 10000
	I0_arr = np.arange(np.max(abs(data)), 10*np.max(abs(data)), ((10*np.max(abs(data))-np.max(abs(data)))/arr_len))
	x0_arr = np.arange(-2*sun_diam_rad, -0.5*sun_diam_rad, 1.5*sun_diam_rad/arr_len)
	y0_arr = np.arange(-2*sun_diam_rad, -0.5*sun_diam_rad, 1.5*sun_diam_rad/arr_len)
	sigx_arr = np.arange(sig_stria, sig_sun, (sig_sun-sig_stria)/arr_len)
	sigy_arr = np.arange(sig_stria, sig_sun, (sig_sun-sig_stria)/arr_len)
	theta_arr = np.arange(0, 2*np.pi, 2*np.pi/arr_len)
	C_arr = np.arange(0,abs(bg_med), abs(bg_med)/arr_len)

	for i in range(niters):
		rand_arr = np.random.randint(0,arr_len,nparams)
		I0g = I0_arr[rand_arr[0]]
		x0g = x0_arr[rand_arr[1]]
		y0g = y0_arr[rand_arr[2]]
		sigxg = sigx_arr[rand_arr[3]]
		sigyg = sigy_arr[rand_arr[4]]
		thetag = theta_arr[rand_arr[5]]
		Cg = 0#C_arr[rand_arr[6]]
		g = func(u,v,I0g, x0g,y0g, sigxg, sigyg, thetag, Cg)
		chisqr = np.sum(weight*abs(data-g)**2)/(len(data)-nparams)
		if i == 0:
			best_params = np.array([I0g, x0g,y0g, sigxg, sigyg, thetag, Cg])
			best_chisqr = chisqr
		elif abs(1-chisqr) < abs(1-best_chisqr):
			best_params = np.array([I0g, x0g,y0g, sigxg, sigyg, thetag, Cg])
			best_chisqr = chisqr

	return best_params, best_chisqr

vis_file = "SB076burst.npz" #"SB076MS_data.npz"
sb = int(vis_file.split("SB")[-1][:3])
epoch_start = datetime(1858,11,17) #MJD

ncorrs=666
int_freqs = np.load("int_freqs.npy")

load_vis = np.load(vis_file)
freq = float(load_vis["freq"])
delt = float(load_vis["dt"])
delf = float(load_vis["df"])
wlen = c.value/freq
uvws = load_vis["uvws"]/wlen
times = load_vis["times"]
data = load_vis["data"]
vis = load_vis["vis"]
weights = load_vis["weights"]
# model = load_vis["model"]
ant0 = load_vis["ant0"]
ant1 = load_vis["ant1"]
# eig = load_vis["eig"] 
"""
eig is largest eigenvalue of gain matrix G == ||G|| 
see https://casa.nrao.edu/casadocs/casa-5.1.0/reference-material/data-weights
"""
t=0
auto_corrs = np.where(ant0==ant1)[0]
cross_corrs = np.where(ant0!=ant1)[0]

# V_mdl0 = np.sum(model,axis=0)
data = data[0,:] + data[3,:]
V = vis[0,:] + vis[3,:]
# V = np.log(V)
vis_err = np.sqrt(1/weights)#(eig[ant0]*eig[ant1])
# weights = 1/(vis[:,auto_corrs[ant0]]*vis[:,auto_corrs[ant1]])
weights = (weights[0,:] + weights[3,:])#/(eig[ant0]*eig[ant1])
#averaged weight is sum of weights according to 
# https://www.astron.nl/lofarwiki/lib/exe/fetch.php?media=public:user_software:documentation:ndppp_weights.pdf (accessed 08/08/2019)
# and maths, I guess

# W_ij = 1/sig_ij**2 = (delt*delf*G_i*G_j)/(v_ii*v_ij)

#vis_err = np.sqrt(1/weights)#np.sqrt((vis[:,auto_corrs[ant0]]*vis[:,auto_corrs[ant1]])/(delt*delf))
vis_err = np.sqrt(abs(vis_err[0,:])**2 + abs(vis_err[3,:])**2)
# weights = ((eig[ant0]*eig[ant1])*(delt*delf))/(vis[0,auto_corrs[ant0]]*vis[0,auto_corrs[ant1]]) +( (eig[ant0]*eig[ant1])*(delt*delf))/(vis[0,auto_corrs[ant0]]*vis[0,auto_corrs[ant1]])
# vis_err = np.log(vis_err)

d_auto = {"u":uvws[0,:][auto_corrs],"v":uvws[1,:][auto_corrs],"w":uvws[2,:][auto_corrs], 
"times":times[auto_corrs], "vis":V[auto_corrs], "vis_err":vis_err[auto_corrs],"weight":weights[auto_corrs]}
d_cross = {"u":uvws[0,:][cross_corrs],"v":uvws[1,:][cross_corrs],"w":uvws[2,:][cross_corrs], 
"times":times[cross_corrs], "vis":V[cross_corrs], "vis_err":vis_err[cross_corrs],"weight":weights[cross_corrs]}

df_auto = pd.DataFrame(data=d_auto) 
df_cross = pd.DataFrame(data=d_cross) 

 

uv_dist = np.sqrt(df_cross.u**2 + df_cross.v**2)

ang_scales = Angle((1/uv_dist)*u.rad)


a0 = np.where(ang_scales.arcmin < 20 )[0]
a1 = np.where(ang_scales.arcmin > 100 )[0]

u0 = np.where(df_cross.u.sort_values() < 200)[0]#np.where(df_cross.u < 500)[0]
u1 = np.where(df_cross.u.sort_values() >-200)[0]#np.where(df_cross.u[u0] > -500)[0]
u_zoom = df_cross.u[df_cross.u.sort_values().index[u1[0]:u0[-1]]] 
v_zoom = df_cross.v[df_cross.u.sort_values().index[u1[0]:u0[-1]]] 

bg = np.where(ang_scales.arcmin < 2 )[0]
bg_vis = df_cross.vis[bg]
bg_med = np.median(bg_vis) 
df_cross = df_cross.assign(vis_log = (np.log(df_cross.vis) - np.log(bg_med)))
df_cross = df_cross.assign(bg_vis = (df_cross.vis - bg_med))
# V_mdl0 = V_mdl0[cross_corrs]

sun_diam_rad = Angle(0.5*u.deg).rad
sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

stria_oom = Angle(0.1*u.arcmin).rad
sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

scatter_diam = Angle(10*u.arcmin).rad
sig_scatter = scatter_diam/(2*np.sqrt(2*np.log(2)))

sig_x_guess = 0.625*sig_sun#Angle(18*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))
sig_y_guess = sig_sun#Angle(7*u.arcmin).rad/(2*np.sqrt(2*np.log(2)))

#params use starting values determined by playing around with it.

# params.add_many(('I0',abs(np.sum(df_cross.vis))/2,True,abs(df_cross.vis).max()), ('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,0),
# 	('y0',sun_diam_rad,True,0,2*sun_diam_rad), ('sig_x',sig_x_guess,True,sig_stria,sig_sun),
# 	('sig_y',sig_y_guess,True,sig_stria,sig_sun), ('theta',np.pi/2,True,0,np.pi))#, ('C', 1, True, 0))
vis_scale = (((df_cross.vis - np.min(abs(df_cross.vis))))/(np.max(abs(df_cross.vis))-np.min(abs(df_cross.vis)))) 
# -1+(((df_cross.vis - np.min(df_cross.vis))*2)/(np.max(df_cross.vis)-np.min(df_cross.vis)))
weight_scale = (df_cross.weight - np.min(abs(df_cross.weight)))/(np.max(abs(df_cross.weight))-np.min(abs(df_cross.weight)))
# vis_scale = df_cross.vis_log
auto_weight = abs((delt*delf)/(data[auto_corrs[ant0]]*data[auto_corrs[ant1]])[cross_corrs])



# mc_params, mc_chi = MC_fit(gauss_2D, vis_scale, df_cross.u, df_cross.v, 1, 1000)
#weight_scale = (auto_weight - np.min(auto_weight))/(np.max(auto_weight)-np.min(auto_weight))
params = Parameters()
params.add_many(('I0',np.max(df_cross.bg_vis.real),True,0,abs(np.max(df_cross.vis))*10), ('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,2*sun_diam_rad),
	('y0',-sun_diam_rad,True,-2*sun_diam_rad,2*sun_diam_rad), ('sig_x',sig_x_guess,True,sig_stria,2*sig_sun),
	('sig_y',sig_y_guess,True,sig_stria,2*sig_sun), ('theta',np.pi/4,True,-np.pi/2,np.pi/2))#, ('C',10*np.min(abs(df_cross.vis)), True, 0, 100*np.min(abs(df_cross.vis))))
# params.add_many(('I0',mc_params[0],True,0), ('x0',mc_params[1],True,-2*sun_diam_rad,-sun_diam_rad/2),
# 	('y0',mc_params[2],True,-2*sun_diam_rad,-sun_diam_rad/2), ('sig_x',mc_params[3],True,sig_stria,sig_sun),
# 	('sig_y',mc_params[4],True,sig_stria,sig_sun), ('theta',mc_params[5],True,0,2*np.pi), ('C', mc_params[6],True))

# params.add_many(('I0',np.sum(abs(df_cross.vis)),True,0,np.sum(abs(df_cross.vis))*10), ('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,-sun_diam_rad/4),
# 	('y0',-sun_diam_rad,True,-1.5*sun_diam_rad,-sun_diam_rad/2), ('sig_x',sig_x_guess,True,sig_stria,0.75*sig_sun),
# 	('sig_y',sig_y_guess,True,sig_stria,sig_sun), ('theta',3*np.pi/4,True,0,2*np.pi), ('C', np.max(abs(df_cross.vis))/1e3,True, np.max(abs(df_cross.vis))/1e4, np.max(abs(df_cross.vis))/100 ))

# params.add_many(('I0',np.max(abs(df_cross.vis))*2,True,0), ('x0',-sun_diam_rad/2,True,-1*sun_diam_rad,-sun_diam_rad/4),
# 	('y0',-sun_diam_rad,True,-2*sun_diam_rad,-sun_diam_rad/2), ('sig_x0',sig_x_guess,True,sig_stria,0.75*sig_sun),
# 	('sig_y0',sig_y_guess,True,sig_stria,sig_sun), ('theta0',1*np.pi/4,True,0,2*np.pi), ('C0', 0,True),
# 	('I1',np.max(abs(df_cross.vis))*2,True,0), ('x1',-sun_diam_rad/2,True,-1*sun_diam_rad,-sun_diam_rad/4),
# 	('y1',-sun_diam_rad,True,-2*sun_diam_rad,-sun_diam_rad/2), ('sig_x1',sig_x_guess,True,sig_stria,0.75*sig_sun),
# 	('sig_y1',sig_y_guess,True,sig_stria,sig_sun), ('theta1',1*np.pi/4,True,0,2*np.pi), ('C1', 0,True))


gmodel = Model(gauss_2D, independent_vars=['u','v']) # Model(two_gauss_V, independent_vars=['u','v'])
result = gmodel.fit(df_cross.vis, u=df_cross.u, v=df_cross.v, params=params, weights=df_cross.weight)#(delt*delf)/(data[auto_corrs[ant0]]*data[auto_corrs[ant1]])[cross_corrs])#df_cross.weight)
pars = result.params 

# ndim, nwalkers = 6, 250
# guess = [*params.valuesdict().values()]
# guess = np.array(guess)
# pos = [guess + guess*1e-4*np.random.randn(ndim) for i in range(nwalkers)]
# with Pool() as ep:
# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_real, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
# 	sampler.run_mcmc(pos,500)

# fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True) 
# samples = sampler.chain 
# labels = ["I0", "x0", "y0", "sig_x","sig_y","theta"] 
# for i in range(ndim): 
# 	ax = axes[i] 
# 	ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 	ax.set_ylabel(labels[i]) 
# 	ax.yaxis.set_label_coords(-0.1, 0.5) 
 
# axes[-1].set_xlabel("step number");

# samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

# mc_pars = np.percentile(samples,[16,50,84],axis=0)
# c_plot = corner.corner(samples, labels=["I0", "x0", "y0", "sig_x","sig_y", "theta"], truths=[*mc_pars[1]]) 


ndim, nwalkers = 4, 250
guess = [*params.valuesdict().values()]
guess.pop(1)
guess.pop(1)
guess = np.array(guess)
pos = [guess + guess*1e-4*np.random.randn(ndim) for i in range(nwalkers)]
with Pool() as ep:
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_abs, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
	sampler.run_mcmc(pos,500)

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True) 
samples = sampler.chain 
labels = ["I0", "sig_x","sig_y","theta"] 
for i in range(ndim): 
	ax = axes[i] 
	ax.plot(sampler.chain[:, :, i].T, "k", alpha=0.3) 
	ax.set_ylabel(labels[i]) 
	ax.yaxis.set_label_coords(-0.1, 0.5) 
 
axes[-1].set_xlabel("step number");
plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_chain_abs.png")

samples = sampler.chain[:, 50:, :].reshape((-1, ndim))

mc_pars = np.percentile(samples,[16,50,84],axis=0)
c_plot = corner.corner(samples, labels=["I0", "sig_x","sig_y", "theta"], truths=[*mc_pars[1]]) 
plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_corner_abs.png")

phs_ndim = 2
phs_params = Parameters()
phs_params.add_many(('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,2*sun_diam_rad),
	('y0',-sun_diam_rad,True,-2*sun_diam_rad,2*sun_diam_rad))
phs_guess = [*phs_params.valuesdict().values()]
phs_guess = np.array(phs_guess)
phs_pos = [phs_guess + phs_guess*1e-4*np.random.randn(phs_ndim) for i in range(nwalkers)]
with Pool() as ep:
	phs_sampler = emcee.EnsembleSampler(nwalkers, phs_ndim, lnprob_real, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
	phs_sampler.run_mcmc(phs_pos,500)

fig, axes = plt.subplots(phs_ndim, figsize=(10, 7), sharex=True) 
phs_samples = phs_sampler.chain 
labels = ["x0","y0"] 
for i in range(phs_ndim): 
	ax = axes[i] 
	ax.plot(phs_sampler.chain[:, :, i].T, "k", alpha=0.3) 
	ax.set_ylabel(labels[i]) 
	ax.yaxis.set_label_coords(-0.1, 0.5) 
 
axes[-1].set_xlabel("step number");
plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_chain_xy.png")

phs_samples = phs_sampler.chain[:, 50:, :].reshape((-1, phs_ndim))

phs_mc_pars = np.percentile(phs_samples,[16,50,84],axis=0)
phs_c_plot = corner.corner(phs_samples, labels=["x0","y0"], truths=[*phs_mc_pars[1]]) 
plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_corner_xy.png")
# tot_ndim = 6
# tot_params = Parameters()
# tot_params.add_many(('I0',np.max(df_cross.bg_vis.real),True,0,abs(np.max(df_cross.vis))*10),
# 	('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,2*sun_diam_rad),
# 	('y0',-sun_diam_rad,True,-2*sun_diam_rad,2*sun_diam_rad),
# 	('sig_x',sig_x_guess,True,sig_stria,2*sig_sun),
# 	('sig_y',sig_y_guess,True,sig_stria,2*sig_sun), 
# 	('theta',np.pi/4,True,-np.pi/2,np.pi/2))

# tot_guess = [*tot_params.valuesdict().values()]
# tot_guess = np.array(tot_guess)
# tot_pos = [tot_guess + tot_guess*1e-4*np.random.randn(tot_ndim) for i in range(nwalkers)]
# with Pool() as ep:
# 	tot_sampler = emcee.EnsembleSampler(nwalkers, tot_ndim, lnprob, args=(df_cross.u,df_cross.v,df_cross.bg_vis,df_cross.weight), pool=ep)
# 	tot_sampler.run_mcmc(tot_pos,500)

# fig, axes = plt.subplots(tot_ndim, figsize=(10, 7), sharex=True) 
# tot_samples = tot_sampler.chain 
# labels = ["I0", "x0", "y0", "sig_x", "sig_y", "theta"] 
# for i in range(tot_ndim): 
# 	ax = axes[i] 
# 	ax.plot(tot_sampler.chain[:, :, i].T, "k", alpha=0.3) 
# 	ax.set_ylabel(labels[i]) 
# 	ax.yaxis.set_label_coords(-0.1, 0.5) 
 
# axes[-1].set_xlabel("step number");
# plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_chain_full.png")

# tot_samples = tot_sampler.chain[:, 50:, :].reshape((-1, tot_ndim))

# tot_mc_pars = np.percentile(tot_samples,[16,50,84],axis=0)
# tot_c_plot = corner.corner(tot_samples, labels=["I0", "x0", "y0", "sig_x", "sig_y", "theta"], truths=[*tot_mc_pars[1]]) 
# plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_corner_full.png")

# mc_pars = np.insert(mc_pars[1],1,0)
# mc_pars = np.insert(mc_pars,1,0)


two_fit_pars = np.insert(mc_pars[1],1,phs_mc_pars[1])
# one_fit_pars = tot_mc_pars[1]
mc_fit_params = Parameters()
mc_fit_params.add_many(('I0',two_fit_pars[0],False), 
	('x0',two_fit_pars[1],True,-2*sun_diam_rad,2*sun_diam_rad),
	('y0',two_fit_pars[2],True,-2*sun_diam_rad,2*sun_diam_rad),
	('sig_x',two_fit_pars[3],False),
	('sig_y',two_fit_pars[4],False), 
	('theta',two_fit_pars[5],False))


emcee_lm_result = gmodel.fit(df_cross.bg_vis, u=df_cross.u, v=df_cross.v, params=mc_fit_params, weights=df_cross.weight,method='emcee')
emcee_lm_pars = emcee_lm_result.params 
print("finished emcee lmfit")
lm_result = gmodel.fit(df_cross.bg_vis, u=df_cross.u, v=df_cross.v, params=mc_fit_params, weights=df_cross.weight)
lm_pars = lm_result.params 
# arr_size = 10000
# u_arr = np.arange(df_cross.u.min(),df_cross.u.max(),(df_cross.u.max()-df_cross.u.min())/arr_size )
# v_arr = np.arange(df_cross.v.min(),df_cross.v.max(),(df_cross.v.max()-df_cross.v.min())/arr_size )
# uv_mesh = np.meshgrid(u_arr,v_arr) 
x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
xy_mesh = np.meshgrid(x_arr,y_arr) 
# mcg = gauss_2D(df_cross.u, df_cross.v, *mc_pars) 
two_fit_I = gauss_I_theta(xy_mesh[0], xy_mesh[1], *two_fit_pars)
# one_fit_I = gauss_I_theta(xy_mesh[0], xy_mesh[1], *one_fit_pars)
# i=1
# for fit_I in [one_fit_I, two_fit_I]:
# 	fig, ax = plt.subplots() 
# 	ax.imshow(fit_I, origin='lower',extent=[Angle(x_arr[0]*u.rad).arcsec, Angle(x_arr[-1]*u.rad).arcsec,
# 		Angle(y_arr[0]*u.rad).arcsec, Angle(y_arr[-1]*u.rad).arcsec])
# 	limb = Circle((0,0),Angle(15*u.arcmin).arcsec, color='r', fill=False)
# 	ax.add_patch(limb)
# 	plt.xlabel("X (arcsec)")
# 	plt.ylabel("Y (arcsec)")
# 	plt.title("mcmc_recreate_{}_fit".format(str(i)))
# 	plt.savefig("/Users/murphp30/Documents/Postgrad/useful_images/mcmc_recreate_{}_fit.png".format(str(i)))
# 	i+=1
# # plt.figure()
# plt.scatter(df_cross.u, df_cross.v,c=abs(df_cross.vis))
# plt.imshow(mcg, aspect="auto", origin="lower", vmin=abs(df_cross.vis).min(),
	# vmax=abs(df_cross.vis).max(), extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])


# gmodel_abs = Model(abs_gauss_2D, independent_vars=['u','v']) # Model(two_gauss_V, independent_vars=['u','v'])
# result_abs = gmodel_abs.fit(abs(df_cross.vis), u=df_cross.u, v=df_cross.v, params=params, weights=df_cross.weight)#(delt*delf)/(data[auto_corrs[ant0]]*data[auto_corrs[ant1]])[cross_corrs])#df_cross.weight)
# pars_abs = result_abs.params 
# emcee_plot = corner.corner(result.flatchain, labels=result.var_names,truths=list(result.params.valuesdict().values())) 
#vis_scale_err,
# plt.figure()
# plt.errorbar(ang_scales.arcminute, abs(df_cross.vis)-abs(bg_med), marker='o',ls='',ecolor='r')
# plt.plot(ang_scales.arcminute, abs(mcg), 'r+',zorder=10)
# # plt.plot(ang_scales.arcminute, abs(result.best_fit), 'r+',zorder=10)
# # plt.plot(ang_scales.arcminute, abs(result.init_fit), 'y+',zorder=10) 
# plt.title("Visibilities vs angular scale")
# plt.xlabel("Angular Scale (arcmin)")
# plt.ylabel("Visibilty power (au)")
# plt.xscale('log')
# plt.savefig("vis_angscale_SB{}_t{}".format(str(int(sb)).zfill(3),str(int(t)).zfill(3)))

# mi = minimize(res_gauss_2D, params,  method='nelder',
# 	args=(df_cross.u, df_cross.v,df_cross.vis,df_cross.weight), nan_policy='omit')        
# # fit = res_gauss_2D(mi.params,df_cross.u,df_cross.v,df_cross.vis,df_cross.weight)
# # f = (fit * df_cross.weight)+df_cross.vis 

# res = minimize(res_gauss_2D, method='emcee', args=(df_cross.u, df_cross.v,df_cross.vis,df_cross.weight), 
# 	nan_policy='omit',burn=300, steps=1000, thin=20, params=mi.params,is_weighted=True)   
# emcee_plot = corner.corner(res.flatchain, labels=res.var_names,truths=list(res.params.valuesdict().values())) 

# plt.figure()
# plt.errorbar(ang_scales.arcminute, np.log(abs(df_cross.vis)),np.log(df_cross.vis_err), marker='o',ls='',ecolor='r')
# plt.plot(ang_scales.arcminute, abs(resultl.best_fit), 'r+',zorder=10)
# # plt.plot(ang_scales.arcminute, abs(result.init_fit), 'y+',zorder=10) 
# plt.title("Visibilities vs angular scale")
# plt.xlabel("Angular Scale (arcmin)")
# plt.ylabel("Visibilty power (au)")
# plt.xscale('log')

# plt.savefig('vis_fit.png')
# params.add_many(('I0',abs(np.sum(df_cross.vis))/2,True,abs(df_cross.vis).max()), ('x0',-sun_diam_rad/2,True,-2*sun_diam_rad,0),
# 	('y0',sun_diam_rad,True,0,2*sun_diam_rad), ('sig_x',sig_x_guess,True,sig_stria,sig_sun),
# 	('sig_y',sig_y_guess,True,sig_stria,sig_sun), ('theta',np.pi/2,True,0,np.pi))#, ('C', 1, True, 0))

# (('I0',0), ('x0',-5e-3,True,-2*sun_diam_rad,2*sun_diam_rad),
# 	('y0',1e-3,True,-2*sun_diam_rad,2*sun_diam_rad), ('sig_x',1e-2,True,0,2*sig_sun),
# 	('sig_y',1e-3,True,0,2*sig_sun), ('theta',0,True,-np.pi,np.pi))

# params2 = Parameters()
# params2.add_many(('I0',abs(df_cross.vis.max())*10,True,0), ('x0',-sun_diam_rad,True,-5*sun_diam_rad,5*sun_diam_rad),
# 	('y0',-sun_diam_rad,True,-5*sun_diam_rad,5*sun_diam_rad), ('sig_x0',sig_stria,True,0,sig_sun),
# 	('sig_y0',sig_stria,True,0,sig_sun), ('theta0',0,True,0,2*np.pi),
# 	('I1',0,True,0), ('x1',0,True,-5*sun_diam_rad,5*sun_diam_rad),
# 	('y1',0,True,-5*sun_diam_rad,5*sun_diam_rad), ('sig_x1',sig_sun,True,0,2*sig_sun),
# 	('sig_y1',sig_sun,True,0,2*sig_sun), ('theta1',0,True,0,2*np.pi)
# 	)
# params2.add_many(('I0',abs(df_cross.vis.max())*10,True,0), ('x0',0),
# 	('y0',0), ('sig_x0',1e-3,True,0,sig_sun),
# 	('sig_y0',1e-3,True,0,sig_sun), ('theta0',0,True,-np.pi,np.pi),
# 	('I1',abs(df_cross.vis.max()),True,0), ('x1',0), 
# 	('y1',0),('sig_x1',1e-3,True,0,2*sig_sun),
# 	('sig_y1',1e-3,True,0,2*sig_sun), ('theta1',0,True,-np.pi,np.pi)
# 	)

# params1 = Parameters()
# params1.add_many(('I0',V_mdl0.max().real,True,0), ('x0',0), ('y0',0), ('sig_x',5e-4,True,0),
 # ('sig_y',1e-3,True,0), ('theta',0))



# resultz = gmodel.fit((df_cross.vis-bg_med)[df_cross.u.sort_values().index[u1[0]:u0[-1]]], 
# 	u=u_zoom, v=v_zoom, params=params,weights=df_cross.weight[df_cross.u.sort_values().index[u1[0]:u0[-1]]])
# parsz = resultz.params 
# gmodel2 = Model(two_gauss_V, independent_vars=['u','v'])
# result2= gmodel2.fit(df_cross.vis, u=df_cross.u, v=df_cross.v, params=params2,weights=df_cross.weight)


# print(result.fit_report()) 
# sig_x = pars['sig_x'].value
# sig_y = pars['sig_y'].value
# fwhmx = FWHM(sig_x)
# fwhmy = FWHM(sig_y)
# print("FWHMX: %.3f arcmin, FWHMY: %.3f arcmin" %(fwhmx,fwhmy))

# pars2 = result2.params
# print(result2.fit_report()) 




# gmodel1 = Model(log_gauss_2D, independent_vars=['u','v'])
# result1 = gmodel1.fit(V_mdl0, u=df_cross.u, v=df_cross.v, params=params1)
# pars1 = result1.params 

# arr_size = 10000
# u_arr = np.arange(df_cross.u.min(),df_cross.u.max(),(df_cross.u.max()-df_cross.u.min())/arr_size )
# v_arr = np.arange(df_cross.v.min(),df_cross.v.max(),(df_cross.v.max()-df_cross.v.min())/arr_size )
# x_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)
# y_arr = np.arange(-2*0.0142739,2*0.0142739,2*1.39e-5)

# uv_mesh = np.meshgrid(u_arr,v_arr) 
# mcg = abs_gauss_2D(uv_mesh[0], uv_mesh[1], *mc_pars) 
# g = gmodel.eval(pars, u=uv_mesh[0], v=uv_mesh[1])
# gu = gmodel1.eval(pars1, u=u_arr, v=np.zeros(len(u_arr)))  
# gv = gmodel1.eval(pars1, u=np.zeros(len(v_arr)), v=v_arr)
# gz = gmodel.eval(parsz, u=uv_mesh[0], v=uv_mesh[1])   
# g1 = gmodel1.eval(pars1, u=uv_mesh[0], v=uv_mesh[1]) 
# g2 = gmodel2.eval(pars2, u=uv_mesh[0], v=uv_mesh[1]) 

# xy_mesh = np.meshgrid(x_arr,y_arr) 
# I_model = Model(gauss_I_theta, independent_vars=['x','y']) #Model(two_gauss_I, independent_vars=['x','y'])
#I_model1 = Model(gauss_I_theta, independent_vars=['x','y'])
#I_model2 = Model(two_gauss_I, independent_vars=['x','y']) 
# I_fit = I_model.eval(pars, x=xy_mesh[0], y=xy_mesh[1])
# I_fit1 = I_model1.eval(pars1, x=xy_mesh[0], y=xy_mesh[1])
# I_fit2 = I_model2.eval(pars2, x=xy_mesh[0], y=xy_mesh[1])


# #
# plt.figure()
# plt.scatter(df_cross.u, df_cross.v,c=(df_cross.vis-bg_med).imag)
# plt.imshow(g.imag, aspect="auto", origin="lower", vmin=(df_cross.vis-bg_med).imag.min(),
# 	vmax=(df_cross.vis-bg_med).imag.max(), extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
# plt.figure()
# plt.errorbar(ang_scales.arcminute, abs(V_mdl0), marker='o',ls='',ecolor='r')
# plt.plot(Angle(1/np.sqrt(u_arr**2),'rad').arcmin, abs(gu), 'r',zorder=10) 
# plt.plot(Angle(1/np.sqrt(v_arr**2),'rad').arcmin, abs(gv), 'r',zorder=10) 
# plt.axvline(x=10,c='cyan')
# plt.axvline(x=5,c='cyan')
# # plt.plot(ang_scales.arcminute, abs(result.init_fit), 'y+',zorder=10) 
# plt.title("Visibilities vs angular scale")
# plt.xlabel("Angular Scale (arcmin)")
# plt.ylabel("Visibilty power (au)")
# plt.xscale('log')



# plt.figure()
# plt.scatter(df_cross.u, df_cross.v,c=abs(df_cross.vis)-abs(bg_med))
# plt.imshow(abs(g), aspect="auto", origin="lower", vmin=(abs(df_cross.vis)-abs(bg_med)).min(),
# 	vmax=(abs(df_cross.vis)-abs(bg_med)).max(), extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])

# fig, ax = plt.subplots() 
# ax.imshow(I_fit, origin='lower',extent=[Angle(x_arr[0]*u.rad).arcsec, Angle(x_arr[-1]*u.rad).arcsec,
# 	Angle(y_arr[0]*u.rad).arcsec, Angle(y_arr[-1]*u.rad).arcsec])
# limb = Circle((0,0),Angle(15*u.arcmin).arcsec, color='r', fill=False)
# ax.add_patch(limb)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(df_cross.u, df_cross.v,df_cross.vis.real,c='r', zorder=10)
# ax.plot_surface(uv_mesh[0], uv_mesh[1], g.real)
ts = time.time()-t0
print("Time to run:", ts)
plt.show()

