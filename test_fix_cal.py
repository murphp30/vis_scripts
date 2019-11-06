#!/usr/bin/env python

# Python code to apply a correction to dodgy parmdb
# mostly copied from plot_solutions_all_stations_v2.py

import lofar.parmdb as lp
import numpy as np
import matplotlib.pyplot as plt

def scaletimes(t):

    t = t-t[0]

    #t = t/3600.

    return t

parmdb = "/mnt/murphp30_data/typeIII_int/L401001_SB320_uv.dppp.MS/instrument"

parmdbmtable = lp.parmdb(parmdb)
soldict = parmdbmtable.getValuesGrid('*')
names = parmdbmtable.getNames()
stationsnames = np.array([name.split(':')[-1] for name in names])
stationsnames = np.unique(stationsnames)
refstationi = 1
Nstat = len(stationsnames)
refstation = stationsnames[refstationi]
times = soldict['Gain:1:1:Real:{s}'.format(s=refstation)]['times']
times = scaletimes(times)

real11_ref = soldict['Gain:1:1:Real:{s}'.format(s=refstation)]['values']
real00_ref = soldict['Gain:0:0:Real:{s}'.format(s=refstation)]['values']
imag11_ref = soldict['Gain:1:1:Imag:{s}'.format(s=refstation)]['values']
imag00_ref = soldict['Gain:0:0:Imag:{s}'.format(s=refstation)]['values']

#for ref in [real11_ref, real00_ref]:#, imag11_ref, imag00_ref]:
#	plt.plot(times, ref)


valscorr00 = real00_ref +1.j*imag00_ref
valscorr11 = real11_ref +1.j*imag11_ref

phase00_ref = np.angle(valscorr00)
phase11_ref = np.angle(valscorr11)

#plt.figure()
#plt.plot(times, phase00_ref)
#plt.plot(times, phase11_ref)


