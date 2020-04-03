#!/usr/bin/env python

#guess at values used to create figure 8b in Krupar et al. 2020

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
freq_arr = np.array((0.52,0.80,1.0,1.2,1.8,2.02,2.04,3.0,4.0,4.8,5.0,7.0,8.0,10.))*u.MHz
freq_err = np.array((0.05,0.05,0.5,0.5,0.5,0.50,0.50,0.5,0.5,0.5,0.5,0.5,0.5,0.5))*u.MHz
eps_arr =  np.array((0.09,0.10,0.11,0.12,0.14,0.15,0.16,0.17,0.18,0.19,0.2,0.21,0.22,0.23))

np.savez("Krupar_fig8.npz",freq_arr, freq_err, eps_arr)

plt.errorbar(freq_arr.value, eps_arr, xerr=freq_err.value, ls='', marker='o', ecolor='r')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$\varepsilon$')
plt.ylabel('Frequency (MHz)')
plt.show()
