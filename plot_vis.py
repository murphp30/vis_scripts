#!/usr/bin/env python
import sys
sys.path.insert(1,'/mnt/murphp30_data/typeIII_int/scripts')
sys.path.insert(1,'/Users/murphp30/murphp30_data/typeIII_int/scripts')
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import dates
from matplotlib.patches import Circle
from datetime import datetime, timedelta
from scipy import interpolate
from scipy.ndimage import interpolation as interp
import scipy.integrate as integrate
import scipy.optimize as opt
from astropy.constants import c, m_e, R_sun, e, eps0, au
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
from lmfit import Model, Parameters, Minimizer,minimize
import corner
import emcee
from multiprocessing import Pool
import time
import warnings
import sunpy
from sunpy.coordinates import sun, frames
from sunpy.map import header_helper
import argparse
from plot_bf import get_data, get_obs_start, oom_source_size, Newkirk_f
import pdb
from icrs_to_helio import icrs_to_helio
parser = argparse.ArgumentParser()
parser.add_argument('--vis_file', dest='vis_file', help='Input visibility file. \
	Must be npz file of the same format as that created by vis_to_npy.py', default='SB076MS_data.npz')
parser.add_argument('--bf_file', dest='bf_file', help='Input HDF5 file containing beamformed data', default='L401005_SAP000_B000_S0_P000_bf.h5')
parser.add_argument('--peak', dest='peak', type=int, help='index for flux peak', default=28)
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
	x0_p, y0_p = rotate_coords(x0,y0,theta)

	V =  np.exp(-2*np.pi*1j*(u*x0+v*y0)) \
	* (((I0/(2*np.pi)) * np.exp(-(((sig_x**2)*((2*np.pi*u_p)**2))/2) - (((sig_y**2)*((2*np.pi*v_p)**2))/2))) + C)
	# np.exp(- ( ((sig_x**2)*((2*np.pi*u_p)**2))/2 + ((sig_x**2)*((2*np.pi*u_p)**2))/2 ))
	# a = ((np.cos(theta)**2*sig_x**2)/2) + ((np.sin(theta)**2*sig_y**2)/2)
	# b = (-(np.sin(2*theta)*sig_x**2)/4) + ((np.sin(2*theta)*sig_y**2)/4)
	# c = ((np.sin(theta)**2*sig_x**2)/2) + ((np.cos(theta)**2*sig_y**2)/2)
	# V = np.exp(-2*np.pi*1j*(u*x0+v*y0)) \
	# * (((I0/(2*np.pi)) * np.exp(-(a*(2*np.pi*u)**2 + 2*b*(2*np.pi*u)*(2*np.pi*v) + c*(2*np.pi*v)**2))) +C)

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

def line_dist(line, point):
	# find perpendicular distance to a line
	a, b, c = line
	x0, y0 = point
	return np.abs(a*x0+b*y0+c)/np.sqrt(a**2+b**2)

def residual(pars, u,v, data, weights=None, ngauss=1, size=True):
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

		model = gauss_2D(u.values,v.values,I0,x0,y0,sig_x,sig_y,theta,C)
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
			resid = (abs(model)) - (abs(data)) #np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)
		else:
			resid = (np.log(abs(model)) - np.log(abs(data))) * weights#((np.real(model) - np.real(data)) + (np.imag(model) - np.imag(data)))*weights##np.sqrt((np.real(model) - np.real(data))**2 + (np.imag(model) - np.imag(data))**2)*weights
			
	else:
		if weights is None:
			resid = np.angle(model) - np.angle(data) # np.log((model.real - data.real)**2) + np.log((model.imag-data.imag)**2) 
		else:
			resid = (np.angle(model)-np.angle(data))*weights #(np.log((model.real - data.real)**2) + np.log((model.imag-data.imag)**2))*weights#(np.angle(model)-np.angle(data))*weights		
	return resid

def lnlike(pars, u,v,vis,weights=None):
	I0,x0,y0,sig_x,sig_y,theta,C = pars
	u_p,v_p  =  rotate_coords(u,v,theta)
	model = gauss_2D(u,v,I0,x0,y0,sig_x,sig_y,theta,C)
	if weights is None:
		inv_sigma2 = np.ones(len(vis))
	else:
		inv_sigma2 = weights 
	

	diff = (np.real(vis) - np.real(model))**2 + (np.imag(vis) - np.imag(model))**2
	return -0.5*(np.sum(diff**2*inv_sigma2 - np.log(2*np.pi*inv_sigma2)))

def lnprior(pars,vis):
	I0,x0,y0,sig_x,sig_y,theta,C = pars

	sun_diam_rad = Angle(0.5*u.deg).rad
	sig_sun = sun_diam_rad/(2*np.sqrt(2*np.log(2)))

	stria_oom = Angle(.1*u.arcmin).rad
	sig_stria = stria_oom/(2*np.sqrt(2*np.log(2)))

	if 0 < I0 < 10*abs(np.max(vis)) and -2*sun_diam_rad < x0 < -0.25*sun_diam_rad and \
	-2*sun_diam_rad < y0 < -0.25*sun_diam_rad and sig_stria < sig_x < 1*sig_sun and sig_stria < sig_y < 1*sig_sun \
	and 0 < theta < np.pi and  np.min(abs(vis)) <  C < np.max(abs(vis)):
		return 0.0
	return -np.inf


def lnprob(pars, u,v,vis,weights=None):
	lp = lnprior(pars,vis)
	if not np.isfinite(lp):
		return -np.inf
	return lp + lnlike(pars, u,v,vis,weights)

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
	theta =-1*theta
	x_p = (x-x0)*np.cos(theta) - (y-y0)*np.sin(theta)
	y_p = (x-x0)*np.sin(theta) + (y-y0)*np.cos(theta)
	
	I = (I0/(2*np.pi*sig_x*sig_y)) * np.exp(-( (x_p**2/(2*sig_x**2)) + (y_p**2/(2*sig_y**2)) ))
	#(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

	return I

def gauss_f(f,I0,f0,sig_f,C):
	
	# a = ((np.cos(theta)**2)/(2*sig_x**2)) + ((np.sin(theta)**2)/(2*sig_y**2))
	# b = -((np.sin(2*theta))/(4*sig_x**2)) + ((np.sin(2*theta))/(4*sig_y**2))
	# c = ((np.sin(theta)**2)/(2*sig_x**2)) + ((np.cos(theta)**2)/(2*sig_y**2))

	G_f = I0 * np.exp(- ((f-f0)**2/(2*sig_f**2))) + C
		#(-(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2)))

	return G_f

"""
Found these somewhere needed to plot Figure 6 from Chrysaphi et al. 2018
r = np.arange(1.75*R_sun.value,2.75*R_sun.value,1e-3*R_sun.value)
tau32_0 = []                                                                                                                                        
tau32_1 = []                                                                                                                                        
tau40_0 = []                                                                                                                                        
tau40_1 = []                                                                                                                                        

# 
4.5e-8 < e**2/h < 7e-8 
for i in range(len(r)):  
    tau32_0.append(integrate.quad(ang_scatter, r[i], 1.5e11,args=(32e6,4.5e-8))[0]) 
    tau32_1.append(integrate.quad(ang_scatter, r[i], 1.5e11,args=(32e6,7e-8))[0])
    tau40_0.append(integrate.quad(ang_scatter, r[i], 1.5e11,args=(40e6,4.5e-8))[0]) 
    tau40_1.append(integrate.quad(ang_scatter, r[i], 1.5e11,args=(40e6,7e-8))[0])

plt.fill_between(r/R_sun.value, tau0, tau1, color='grey')                                                                                       


plt.yscale("log")                                                                                                                               

plt.ylabel("Optical Depth w.r.t. Scattering")                                                                                                   


plt.xlabel("Heliocentric Distance (R_sun)")                                                                                                     


plt.axhline(y=1, ls='--', color='k')                                                                                                            
                                                                                                           

plt.axvline(x=R.value/R_sun.value, ls='-', color='k')   
"""

def Newkirk(r): 
	n_0 = 4.2e4 
	n = n_0*10**(4.32*R_sun.value/r) #in cm^-3 
	return n*1e6 #assume on streamer axis Riddle 1974, Newkirk 1961

def Saito(r):
	n = 1.36e6 * (R_sun.value/r)**2.14 + 1.86e8 * (R_sun.value/r)**6.13
	return n

def Leblanc(r):
	n = 3.3e5 * (R_sun.value/r)**2 + 4.1e6 * (R_sun.value/r)**4 + 8e7 * (R_sun.value/r)**6
	return n

def Allen_Baumbach(r):
	n = 1e8*(1.55*(R_sun.value/r)**6 + 2.99*(R_sun.value/r)**16) + 4e5*(R_sun.value/r)**2
	return n


def density_3_pl(r):
	#Three power law density used by Kontar et al. 2019
	n = (4.8e9*((R_sun.value/r)**14)) +( 3e8*((R_sun.value/r)**6) )+ (1.4e6*((R_sun.value/r)**2.3))
	return n*1e6
def plasma_freq(r):
	return np.sqrt((e.value**2*density_3_pl(r))/(eps0.value*m_e.value))/(2*np.pi)

def ang_scatter(r,freq,e_sq_over_h): 
	#e_sq_over_h = 5e-8 # m^-1 Steinberg et al. 1971, Riddle 1974, Chrysaphi et al. 2018 
	f_p = plasma_freq(r)
	return (np.sqrt(np.pi)/2) * ((f_p**4)/((freq**2-f_p**2)**2) )* e_sq_over_h 

def angle_integral(r, freq):
	f_p = plasma_freq(r)
	return (np.sqrt(np.pi)/2)*((f_p**4)/((freq**2-f_p**2)**2)) * (r**3/2)

def freq_integral(r, freq, scale):
	f_p = plasma_freq(r)
	#Krupar 2018 10.3847/1538-4357/aab60f
	if r < 100*R_sun.value:
		li = (684/np.sqrt(density_3_pl(r)*1e-6))*1e3#(r/R_sun.value)*1e3
	else:
		li = 100*1e3 
	#Wohlmuth 2001 https://link-springer-com.elib.tcd.ie/content/pdf/10.1023/A:1011845221808.pdf
	lo = 0.25*R_sun.value * (r/R_sun.value)**0.82
	if scale == "krup":
		h = li**(1/3) * lo**(2/3)
	elif scale == "stein":
		h = 5e-5 * r
	else:
		print("Invalid scale: {}".format(scale))
	return (np.sqrt(np.pi)/(2*h))*((f_p**4)/((freq**2-f_p**2)**2))

def dist_from_freq(freq): 
	kappa = np.sqrt((e.value**2/(m_e.value*eps0.value)))/(2*np.pi) 
	n_0 = 4.2e4 * 1e6 #cm^-3 to m^-3, keeping it SI 
	r = R_sun.value * (2.16)/(np.log10(freq)-np.log10(kappa*np.sqrt(n_0))) 
	return r 

def stria_fit(i, save=False):
	striae = bf_data[int(np.round(burst_delt*(i-q_t))),:]
	striae_bg = np.mean(striae[-500:])
	bf_params = Parameters()
	bf_params.add_many(('I0', striae[burst_f_mid_bf], True, 0),
		('f0', bf_freq[burst_f_mid_bf], True, bf_freq[burst_f_mid_bf-8], bf_freq[burst_f_mid_bf+8]),
		('sig_f', (bf_delf*4)/(2*np.sqrt(2*np.log(2))), True, 0, (bf_delf*32)/(2*np.sqrt(2*np.log(2)))),
		('C', striae_bg, True, 0, striae_bg*1.1 )
		)

	bf_model = Model(gauss_f)
	bf_range = striae[burst_f_mid_bf-16:burst_f_mid_bf+16]
	freq_range = bf_freq[burst_f_mid_bf-16:burst_f_mid_bf+16]
	bf_result = bf_model.fit(bf_range, bf_params, f=freq_range)
	best_pars = [par.value for par in [*bf_result.params.values()]]
	g_f = gauss_f(bf_freq[burst_f_mid_bf-100:burst_f_mid_bf+100], *best_pars)
	
	plt.figure()
	plt.plot(bf_freq[burst_f_mid_bf-100:burst_f_mid_bf+100], striae[burst_f_mid_bf-100:burst_f_mid_bf+100])
	plt.plot(bf_freq[burst_f_mid_bf-100:burst_f_mid_bf+100],g_f)
	plt.title("Striae Intensity")
	plt.ylabel("Intensity (above background)")
	plt.xlabel("Frequency (MHz)")
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/stria_fit_t{1}.png".format(str(SB).zfill(3),str(i-q_t).zfill(3)))
		plt.close()
	return bf_result

""" 
Following two functions "borrowed" from dft_acc.py
Original by Menno Norden, James Anderson, Griffin Foster, et al.
"""

def dft2(d,k,l,u,v):
    """compute the 2d DFT for position (k,l) based on (d,uvw)"""
    return np.sum(d*np.exp(-2.*np.pi*1j*((u*k) + (v*l))))
    #psf:
    #return np.sum(np.exp(-2.*np.pi*1j*((u*k) + (v*l))))
def dftImage(d,u,v,px,res,mask=False):
    """return a DFT image"""
    nants=(1+np.sqrt(1+8*d.shape[0]))/2
    im=np.zeros((px[0],px[1]),dtype='complex64')
    mid_k=int(px[0]/2.)
    mid_l=int(px[1]/2.)
    # u=uvw[:,:,0]
    # v=uvw[:,:,1]
    # w=uvw[:,:,2]
    u_new = u/mid_k
    v_new = v/mid_l
    start_time=time.time()
    for k in range(px[0]):
        for l in range(px[1]):
            im[k,l]=dft2(d,(k-mid_k),(l-mid_l),u_new,v_new)
            if mask:        #mask out region beyond field of view
                rad=(((k-mid_k)*res)**2 + ((l-mid_l)*res)**2)**.5
                if rad > mid_k*res: im[k,l]=0
                #else: im[k,l]=dft2(d,(k-mid_k),(l-mid_l),u,v)
    print(time.time()-start_time)
    #pdb.set_trace()
    return im

# def par_dft2(k,l):
#     return dft2(1, res*(k-mid_k), res*(l-mid_l), burst.u, burst.v)
# with Pool() as p:  
#     start_time=time.time()
#     im = p.starmap(par_dft2, product(reversed(range(px[0])),range(px[1])))
#     p.close()
#     p.join()
#     print(time.time()-start_time)
#     im = np.array(im).reshape(px[0], px[1])
#     plt.imshow(abs(im), origin='lower', aspect='equal')
#     plt.show()

def make_map(vis, data,dpix):
	#make sunpy map from LOFAR_vis meta data 
	#and reconstructed image data
	icrs_dict = sunpy.util.MetaDict()
	icrs_dict['crpix1'] = (data.shape[0] -1) /2
	icrs_dict['crpix2'] = (data.shape[1] -1) /2
	icrs_dict['cdelt1'] = Angle(dpix*u.rad).deg
	icrs_dict['cdelt2'] = Angle(dpix*u.rad).deg
	icrs_dict['cunit1'] = 'deg'
	icrs_dict['cunit2'] = 'deg'
	icrs_dict['crval1'] = vis.phs_dir.ra.deg
	icrs_dict['crval2'] = vis.phs_dir.dec.deg
	icrs_dict['crval3'] = vis.freq
	icrs_dict['ctype1'] = 'RA---SIN'
	icrs_dict['ctype2'] = 'DEC--SIN'
	icrs_dict['date-obs'] = vis.time.isoformat() #strftime("%Y-%m-%dT%H:%M:%S.%f")

	icrs_map = sunpy.map.Map(data, icrs_dict)
	# phs_dir_helio = vis.phs_dir.transform_to(frame='helioprojective', merge_attributes=True)
	# # shift_x = 0#vis.solar_ra_offset.to(u.rad)/Angle(dpix*u.rad)
	# # shift_y = 0#vis.solar_dec_offset.to(u.rad)/Angle(dpix*u.rad)
	# # data = interp.shift(data, (shift_x, -shift_y))
	# data = interp.rotate(data, vis.solar_angle.deg)
	# #phs_dir_pix =  data.shape
	# # icrs_head = header_helper.make_fitswcs_header(data, vis.phs_dir)
	# # icrs_smap = sunpy.map.Map(data, icrs_head)
	# helio_scale = Angle(dpix*u.rad).to(u.arcsec)/(1*u.pix)
	# helio_head = header_helper.make_fitswcs_header(data, phs_dir_helio, scale=u.Quantity([helio_scale,helio_scale]))


	return icrs_map #sunpy.map.Map(data,helio_head)

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

		phs_dir = Angle(self.load_vis["phs_dir"]*u.rad)
		

		self.delt = float(self.load_vis["dt"])
		self.delf = float(self.load_vis["df"])
		self.wlen = c.value/self.freq
		self.obsstart = epoch_start + timedelta(seconds=self.load_vis["times"][0][0])
		self.obsend = epoch_start + timedelta(seconds=self.load_vis["times"][-1][0])

		self.time = epoch_start + timedelta(seconds=self.load_vis["times"][self.t][0])


		self.dsun = sun.earth_distance(self.time)
		self.phs_dir =  SkyCoord(phs_dir[0], phs_dir[1],
								distance=self.dsun, obstime=self.time, 
								frame="icrs", equinox="J2000")
		sun_dir =  sun.sky_position(self.time,False)
		self.sun_dir = SkyCoord(*sun_dir)
		self.solar_ra_offset = (self.phs_dir.ra-self.sun_dir.ra)
		self.solar_dec_offset = (self.phs_dir.dec-self.sun_dir.dec)
		self.solar_rad = sun.angular_radius(self.time)
		self.solar_angle = sunpy.coordinates.sun.P(self.time)
	
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
		weights = weights[0,:] + weights[3,:] #1/((1/weights[0,:])+(1/weights[3,:]))		
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
	
	def model_df(self):
		ant0 = self.load_vis["ant0"]
		ant1 = self.load_vis["ant1"]
		#auto_corrs = np.where(ant0==ant1)[0]
		

		uvws = self.load_vis["uvws"]/self.wlen
		times = self.load_vis["times"]
		times = epoch_start + timedelta(seconds=1)*times
		data = self.load_vis["data"]
		vis = self.load_vis["mdl"]
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

		q_t = 1199//2 #time index before burst, first 10 minutes of data is queit sun ~ 1199 time samples
		#There's actually a burst in the "quiet sun" part of the observation so just halve the time.
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

fov = Angle(1910*5.2338*u.arcsec).rad
px = [258,258]
res = fov/px[0]

x_guess = Angle(11*u.arcmin).rad
y_guess = Angle(12*u.arcmin).rad
sig_x_guess = x_guess/(2*np.sqrt(2*np.log(2)))
sig_y_guess = y_guess/(2*np.sqrt(2*np.log(2)))

x1_guess = Angle(5*u.arcmin).rad
y1_guess = Angle(5*u.arcmin).rad
sig_x1_guess = x1_guess/(2*np.sqrt(2*np.log(2)))
sig_y1_guess = y1_guess/(2*np.sqrt(2*np.log(2)))

vis0 = LOFAR_vis(vis_file, q_t) #first important visibility
q_sun = vis0.queit_sun_df()
arr_size = 5000
u_arr = np.arange(q_sun.u.min(),q_sun.u.max(),(q_sun.u.max()-q_sun.u.min())/arr_size )
v_arr = np.arange(q_sun.v.min(),q_sun.v.max(),(q_sun.v.max()-q_sun.v.min())/arr_size )

net_size = 100
u_net = np.arange(q_sun.u.min(),q_sun.u.max(),net_size)
v_net = np.arange(q_sun.v.min(),q_sun.v.max(),net_size)


uv_mesh = np.meshgrid(u_arr,v_arr)
dpix =  1.39e-5
x_arr = np.arange(-0.0142739,0.0142739,dpix)
y_arr = np.arange(-0.0142739,0.0142739,dpix)
xy_mesh = np.meshgrid(x_arr,y_arr) 

ang_arr = np.arange(0, 600, 600/arr_size)

bg_data = get_data(bf_file, vis0.obsstart, vis0.time )[0]
bg_data = bg_data[:bg_data.shape[0]//2,:]
bg_mean = np.mean(bg_data, axis=0) 
bf_data, bf_freq, bf_tarr = get_data(bf_file, vis0.time, vis0.obsend )
bf_data = bf_data/bg_mean
bf_delt = bf_tarr[1] - bf_tarr[0]
bf_delf = bf_freq[1] - bf_freq[0]

burst_delt = (vis0.delt)/bf_delt
burst_f_mid_bf = np.where(bf_freq == vis0.freq*1e-6 +(bf_delf/2))[0][0]
burst_f_start_bf = burst_f_mid_bf - 8
burst_f_end_bf = burst_f_mid_bf + 8

bf_data_t = np.mean(bf_data[:,burst_f_start_bf:burst_f_end_bf],axis=1)
day_start = get_obs_start(bf_file)
day_start = datetime.strptime(day_start.decode("utf-8"),"%Y-%m-%dT%H:%M:%S.%f000Z")
#equivalent to day_start = datetime(2015,10,17,8,00,00)

bf_dt_arr = day_start + timedelta(seconds=1)*bf_tarr
bf_dt_arr = dates.date2num(bf_dt_arr)
date_format = dates.DateFormatter("%H:%M:%S")

# burst_max_t = burst_delt*q_t+np.argmax(bf_data_t) 


def parallel_fit(i):
	save = False
	model = False
	vis = LOFAR_vis(vis_file, i)
	burst = vis.vis_df()
	ngauss = 1

	params = Parameters()
	if model:
		fit_vis = vis.model_df().vis
		fit_weight = None
	else:
		fit_vis = burst.vis #- q_sun.vis
		uv_grid, _, _ = np.histogram2d(burst.u, burst.v, bins=[u_net, v_net], density=False)
		box_weight = np.zeros(len(burst.u))
		for i in range(len(burst.u)):
			u_box = np.where(abs(u_net - burst.u[i]) == np.min(abs(u_net - burst.u[i])))[0]-1
			v_box = np.where(abs(v_net - burst.v[i]) == np.min(abs(v_net - burst.v[i])))[0]-1
			box_weight[i] = uv_grid[u_box,v_box]+1
		fit_weight = box_weight/np.max(box_weight)#np.ones(len(burst.vis))#burst.weight#np.sqrt(burst.weight**2 + q_sun.weight**2)

	if ngauss == 2:
		params.add_many(('I0',np.pi*np.max(abs(fit_vis)),True,np.pi*np.max(abs(fit_vis)),abs(np.max(fit_vis))*100), 
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
		params.add_many(('I0',2*np.pi*abs(np.max(fit_vis)),True,0), 
			('x0',-0.7*sun_diam_rad,True,-1.5*sun_diam_rad,0),
			('y0',0.5*sun_diam_rad,True,0,1.5*sun_diam_rad), 
			('sig_x',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta',np.pi/3,True,0, np.pi),
			('C',np.mean(abs(fit_vis)),True, np.min(abs(fit_vis))))
		""" works for SB076
		params.add_many(('I0',2*np.pi*abs(np.sum(fit_vis)),True,0), 
			('x0',-0.7*sun_diam_rad,False,-1.5*sun_diam_rad,1.5*sun_diam_rad),
			('y0',0.5*sun_diam_rad,False,-1.5*sun_diam_rad,1.5*sun_diam_rad), 
			('sig_x',sig_x_guess,True,sig_stria,1.5*sig_sun),
			('sig_y',sig_y_guess,True,sig_stria,1.5*sig_sun), 
			('theta',np.pi/4,True,0, np.pi),
			('C',np.mean(abs(fit_vis)),True, np.min(abs(fit_vis))))
		"""

	fit = minimize(residual, params, method="leastsq", args=(burst.u, burst.v, fit_vis, fit_weight, ngauss, True))
	# print("Fitting", i-q_t)
	size_fit_errs = {par+"_err":fit.params[par].stderr for par in fit.params}

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

	# only fit phase to core stations, same clock and all that.
	fit = minimize(residual, fit.params, method="emcee", args=(burst.u[:275], burst.v[:275], fit_vis[:275],fit_weight[:275] ,ngauss,False))
	
	# pos = np.array([fit.params[key].value*(1 + 1e-4*np.random.randn(200)) for key in fit.var_names]).T #np.zeros((200,7)) #because we want 200 walkers for 7 parameters
	# nwalkers, ndim = pos.shape
	# with Pool() as p:
	# 	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=p, args=(burst.u, burst.v, fit_vis, fit_weight))
	# 	sampler.run_mcmc(pos, 1500, progress=True)
	# 	p.join()
	#	p.close()
	
	val_dict = fit.params.valuesdict()

	# val_dict_pos = fit_pos.params.valuesdict()
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
		slope = np.tan((np.pi/2)-val_dict['theta'])
		u_arrp = slope*u_arr 
		v_arrp = (-1/slope)*u_arr
		perp_dist_u = line_dist((slope, -1, 0),(burst.u, burst.v))
		perp_dist_v = line_dist((-1/slope, -1, 0),(burst.u, burst.v))
		min_dist = 10 #minimum distance to line to count as "along the axis"
		u_p = burst.u[np.where(perp_dist_u < min_dist)[0]]
		v_p = burst.v[np.where(perp_dist_v < min_dist)[0]]
		g_fitu = gauss_2D(u_arr, u_arrp, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'], val_dict['C'])
		g_fitv = gauss_2D(u_arr, v_arrp, val_dict['I0'], val_dict['x0'], val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'], val_dict['C'])		

		ang_u = Angle((1/np.sqrt(u_arr**2+u_arrp**2))*u.rad).arcmin
		ang_v = Angle((1/np.sqrt(u_arr**2+v_arrp**2))*u.rad).arcmin
		# u_rot, v_rot = rotate_coords(u_arr, v_arr, val_dict['theta'])
		# g_fitx = ((val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_x']**2 * (2*np.pi*u_rot)**2))/2)) + val_dict['C']
		# g_fity = ((val_dict['I0']/(2*np.pi)) * np.exp(-((val_dict['sig_y']**2 * (2*np.pi*v_rot)**2))/2)) + val_dict['C']
		# ang_u, ang_v = Angle((1/abs(u_rot))*u.rad).arcmin, Angle((1/abs(v_rot))*u.rad).arcamin
		cont_fit = gauss_2D(uv_mesh[0], uv_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'],
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'], val_dict['C'])
		I_fit = gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], -1*val_dict['y0'], 
			val_dict['sig_x'], val_dict['sig_y'], val_dict['theta']) 
		

		"""
		Something weird with y position in that it should be negative but it's not... unless wsclean is wrong?
		Also, which way does python rotate things and which direction does one rotate in real space vs fourier space?
		"""
	# plt.figure()
	fig, ax = plt.subplots()#s = plt.subplots(2,1,gridspec_kw={'height_ratios': [2, 1]},figsize=(8,7))
	ax.plot(burst.ang_scales, (abs(fit_vis)),'o')
	#ax.plot(burst.ang_scales, (abs(g_fit)),'r+') 
	# if ngauss == 1:
	ax.plot(ang_u, (abs(g_fitu)), 'r')
	ax.plot(ang_v, (abs(g_fitv)), 'r')
	# else:
		# ax.plot(burst.ang_scales, (abs(g_fit)),'r+')
	ax.set_xlabel("Angular Scale (arcminute)")
	ax.set_ylabel("Visibility (arbitrary)")
	ax.set_title("Vis vs ang scale {}".format(vis.time.isoformat()))

	ax.set_xscale('log')
	ax.set_xlim([ax.get_xlim()[0], 1e3])
	# if vis_file[2:5] == "076":
	# 	if not model:
	# 		ax.set_xlim([ax.get_xlim()[0], 1e3])
	# 		ax.set_ylim([ax.get_ylim()[0], 0.9e7])

	# axs[1].plot(bf_dt_arr,bf_data_t)
	# axs[1].xaxis_date() 
	# axs[1].xaxis.set_major_formatter(date_format)
	# axs[1].set_xlabel("Time")
	# axs[1].set_ylabel("Intensity (above background)")
	# axs[1].set_title("Intensity vs Time at {}MHz".format(np.round(vis.freq/1e6,2)))
	# axs[1].vlines(bf_dt_arr[int(np.round(burst_delt*(i-q_t)))], ymin=0, ymax=bf_data_t[int(np.round(burst_delt*(i-q_t)))],color='grey')
	plt.tight_layout()
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/vis_ang_scale_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/vis_ang_scale_raw.png")
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
	# 	plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/uv_plane_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
	# 	# plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/uv_raw.png")
	# 	plt.close()

	plt.figure()
	plt.scatter(burst.u, burst.v, c=np.log(abs(fit_vis)))
	# plt.imshow(np.log(abs(cont_fit)), aspect='auto', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]],
	#  vmin=np.min(np.log(abs(fit_vis))), vmax=np.max(np.log(abs(fit_vis))))
	# plt.contour(uv_mesh[0], uv_mesh[1], np.log(abs(cont_fit)), 
	# 	[np.log(0.1) + np.max(np.log(abs(cont_fit))),np.log(0.5) + np.max(np.log(abs(cont_fit))),
	# 	np.log(0.9) + np.max(np.log(abs(cont_fit))),np.log(0.95) + np.max(np.log(abs(cont_fit)))],
	# 	colors='r')
	plt.contour(uv_mesh[0], uv_mesh[1], np.log(abs(cont_fit)), 
	[0.5*np.max(np.log(abs(cont_fit)))],#,0.5*np.max((abs(cont_fit))),
	# 0.9*np.max((abs(cont_fit))),0.95*np.max((abs(cont_fit)))],
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
		plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/uv_zoom_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/uv_cont_raw.png")
		plt.close()
	# plt.figure()
	# plt.imshow(np.log(abs(cont_fit)), aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
	# plt.xlim(-1000,1000)
	# plt.ylim(-1000,1000)
	# # plt.colorbar()
	# fig, ax = plt.subplots()
	# im = ax.imshow(np.flip(I_fit,0), aspect='equal', origin='lower', vmin=0, vmax=3.5e12,
	# 	extent=[Angle(x_arr[0],unit='rad').arcsec, Angle(x_arr[-1],unit='rad').arcsec,
	# 	Angle(y_arr[0],unit='rad').arcsec, Angle(y_arr[-1],unit='rad').arcsec])
	# s = Circle((vis.solar_ra_offset.arcsec,vis.solar_dec_offset.arcsec),vis.solar_rad.arcsec, color='r', fill=False)
	# ax.add_patch(s)
	# plt.xlabel("X (arcsecond)")
	# plt.ylabel("Y (arcsecond)")
	# fig.colorbar(im)
	# plt.title("Recreated Image {}".format(vis.time.isoformat()))
	# plt.tight_layout()
	plt.figure()
	icrs_map = make_map(vis, I_fit,dpix) #make_map(vis, np.flip(I_fit, 0),dpix)
	helio_map = icrs_to_helio(icrs_map)
	helio_map.plot(cmap="viridis", title="Recreated Image {}".format(helio_map.date.datetime.isoformat()))
	helio_map.draw_limb(color='r')
	if save:
		plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/im_recreate_t{1}.png".format(str(SB).zfill(3),str(vis.t-q_t).zfill(3)))
		# plt.savefig("/mnt/murphp30_data/typeIII_int/gain_corrections/vis/im_recreate_raw.png")
		plt.close()
	return fit, size_fit_errs
	#fig, ax = plt.subplots()
	#ax.imshow(abs(gm_fit), aspect='equal', origin='lower', extent=[u_arr[0], u_arr[-1], v_arr[0], v_arr[-1]])
	# s = Circle((0,0),1/vis.solar_rad.rad, color='r', fill=False)
	# ax.add_patch(s)

if __name__ == "__main__":	
	# np.save("/mnt/murphp30_data/typeIII_int/mcmc/SB076/pars_list.npy", pars_list)
	# with Pool() as p_fit:
	# 	fits = p_fit.map(parallel_fit, range(q_t+16, q_t+31))

	"""
	The following is a very lame method of finding the correct time index
	It should really be some f = at^-b relation or something, Peijin probably has
	somehting in his paper but until then...
	Reid & Kontar 2018 t = 1.5f^-0.77 per 30MHz
	"""
	peak_dict = {"059":38, "076":28, "117":50, "118":50, "119":50, "120":50, "125":49, "126":50, "127":50, "130":49, "133":47, "160":23}
	e_krup_list = []
	e_stein_list = []
	freq_list = []
	R_obs_list = []
	R_list = []
	x_list = []
	y_list = []
	fit_pos_list = []
	fit_err_list = []
	# for sb in ["059", "076", "119", "120", "125", "126", "130", "133", "160"]:
	# 	vis_file = vis_file.replace(vis_file[2:5],sb)
	# 	vis0 = LOFAR_vis(vis_file, q_t) #first important visibility
	# 	q_sun = vis0.queit_sun_df()

	# 	arr_size = 5000
	# 	u_arr = np.arange(q_sun.u.min(),q_sun.u.max(),(q_sun.u.max()-q_sun.u.min())/arr_size )
	# 	v_arr = np.arange(q_sun.v.min(),q_sun.v.max(),(q_sun.v.max()-q_sun.v.min())/arr_size )
	# 	uv_mesh = np.meshgrid(u_arr,v_arr)
	# 	dpix =  1.39e-5
	# 	x_arr = np.arange(-0.0142739,0.0142739,dpix)
	# 	y_arr = np.arange(-0.0142739,0.0142739,dpix)
	# 	xy_mesh = np.meshgrid(x_arr,y_arr) 

	# 	ang_arr = np.arange(0, 600, 600/arr_size)

	# 	bg_data = get_data(bf_file, vis0.obsstart, vis0.time )[0]
	# 	bg_data = bg_data[:bg_data.shape[0]//2,:]
	# 	bg_mean = np.mean(bg_data, axis=0) 
	# 	bf_data, bf_freq, bf_tarr = get_data(bf_file, vis0.time, vis0.obsend )
	# 	bf_data = bf_data/bg_mean
	# 	bf_delt = bf_tarr[1] - bf_tarr[0]
	# 	bf_delf = bf_freq[1] - bf_freq[0]

	# 	burst_delt = (vis0.delt)/bf_delt
	# 	burst_f_mid_bf = np.where(bf_freq == vis0.freq*1e-6 +(bf_delf/2))[0][0]
	# 	burst_f_start_bf = burst_f_mid_bf - 8
	# 	burst_f_end_bf = burst_f_mid_bf + 8

	# 	bf_data_t = np.mean(bf_data[:,burst_f_start_bf:burst_f_end_bf],axis=1)
	# 	day_start = get_obs_start(bf_file)
	# 	day_start = datetime.strptime(day_start.decode("utf-8"),"%Y-%m-%dT%H:%M:%S.%f000Z")
	# 	#equivalent to day_start = datetime(2015,10,17,8,00,00)

	# 	bf_dt_arr = day_start + timedelta(seconds=1)*bf_tarr
	# 	bf_dt_arr = dates.date2num(bf_dt_arr)
	# 	date_format = dates.DateFormatter("%H:%M:%S")	
	try:
		peak = peak_dict[vis_file[2:5]]
	except KeyError:
		peak = args.peak
	print("Running subband {}".format(vis_file[2:5]))
	fit_pos,fit_err = parallel_fit(q_t+peak)
	fit_err["x0_err"] = fit_pos.params["x0"].stderr
	fit_err["y0_err"] = fit_pos.params["y0"].stderr
	fit_stria = stria_fit(q_t+peak)
	plt.close()
	t_run = time.time()-t0

	del_f = 2*np.sqrt(2*np.log(2))*fit_stria.params['sig_f'].value
	freq0 = fit_stria.params['f0'].value
	oom = oom_source_size(del_f, freq0)

	print("Time to run:", t_run)
	print("FWHM x: {} arcmin".format(FWHM(fit_pos.params['sig_x'])))
	print("FWHM y: {} arcmin".format(FWHM(fit_pos.params['sig_y'])))
	print("Order of Magnitude size: {} arcmin".format((oom.value/R_sun.value)*vis0.solar_rad.arcmin) )

	# FWHM_x = FWHM(fit_pos.params['sig_x'])#*u.arcmin)
	# FWHM_y = FWHM(fit_pos.params['sig_y'])#*u.arcmin)
	# FWHM_x, FWHM_y = (R_sun.value/15)*FWHM_x, (R_sun.value/15)*FWHM_y
	# R_s =  dist_from_freq(vis0.freq) #Newkirk_f(vis0.freq*1e-6)*R_sun.value

	# area_0 = np.pi*((oom.value/R_sun.value)*Angle(15*u.arcmin).rad)**2
	# area_1 = np.pi*(FWHM_x*FWHM_y)
	# 720/pi made sense at some point I'm sure. Conversion from rad to arcmin?
	x_m = (720/np.pi) * R_sun.value * fit_pos.params['x0'].value
	y_m = (720/np.pi) * R_sun.value * fit_pos.params['y0'].value
	R_obs = np.sqrt(x_m**2 + y_m**2)
	R = R_obs/np.sin(Angle(70*u.deg)) * u.m #from rough estimate of 3d pfss

	# r_arr = np.arange(R.value,au.value, au.value*1e-6)
	# theta = np.arctan(0.5*FWHM_x/(R_obs-R_s))
	# # r = np.arange(Newkirk_f(vis0.freq), R_m, 1e-3)
	# tau = integrate.quad(freq_integral,R_s, R_obs,args=(vis0.freq))[0]
	# for i in range(len(r)-1):
	# 	tau.append(integrate.quad(freq_integral, r[i], r[i+1],args=(vis0.freq))[0])
	freq_int_stein = integrate.quad(freq_integral, R.value, au.value ,args=(vis0.freq,"stein"))[0]
	freq_int_krup = integrate.quad(freq_integral, R.value, au.value ,args=(vis0.freq,"krup"))[0]  
	#e_sq_over_h = 1/freq_int #(2/np.sqrt(np.pi))*(theta/tau)#((area_1-area_0)/np.sum(tau))
	# h = 5e-5*(R.value)
	# li = (R/R_sun)*1e3*u.m  
	# lo = 0.25*R_sun * (R/R_sun)**0.82 
	# l = li**(1/3) * lo**(2/3)
	e_sq_stein = 1/freq_int_stein# * h #e_sq_over_h * h
	e_sq_krup = 1/freq_int_krup# * l.value
	e_stein = np.sqrt(e_sq_stein)
	e_krup = np.sqrt(e_sq_krup)
	e_krup_list.append(e_krup)
	e_stein_list.append(e_stein)
	freq_list.append(vis0.freq*1e-6)
	R_obs_list.append(R_obs)
	R_list.append(R.value)
	x_list.append(x_m/R_sun.value)
	y_list.append(y_m/R_sun.value)
	fit_pos_list.append(fit_pos)
	fit_err_list.append(fit_err)
		# val_dict = fit_pos.params.valuesdict()
	# I_fit = gauss_I(xy_mesh[0], xy_mesh[1], val_dict['I0'], val_dict['x0'], val_dict['y0'], 
	# 		val_dict['sig_x'], val_dict['sig_y'], val_dict['theta'])


	plt.show()

