#!/usr/bin/env python

#convert visibility data to np arrays
#must be run inside CASA
#execfile(vis_to_npy.py)
#therefore python 2

import numpy as np
import argparse
from datetime import datetime, timedelta
"""
parser = argparse.ArgumentParser()
parser.add_argument("SB", default="L401003_SB076_uv.dppp.MS/", type=str)
parser.add_argument("-start_time",type=str, dest="t_start",help="must be of format YYYY-MM-DDTHH:MM:SS", default="2015-10-17T13:20:00")
parser.add_argument("-end_time",type=str, dest="t_end", help="must be of format YYYY-MM-DDTHH:MM:SS",default="2015-10-17T13:21:00")

args = parser.parse_args()
SB = args.SB
t_start = args.t_start
t_end = args.t_end
"""
def get_MS_data(SB,t_start, t_end):
	nbaselines = 666 #find this from the file somewhere
	epoch_start = datetime(1858,11,17)
	tb.open(SB)
	times = tb.getcol("TIME",0,1)
	dt = tb.getcol("EXPOSURE", 0,1)#timedelta(seconds=times[666]-times[0])
	dt_obs_start = epoch_start+timedelta(seconds=times[0])
	dt_start = datetime.strptime(t_start, "%Y-%m-%dT%H:%M:%S.%f")
	dt_end = datetime.strptime(t_end, "%Y-%m-%dT%H:%M:%S.%f")

	i = int(np.round((dt_start-dt_obs_start).total_seconds()/dt))*nbaselines
	no_intervals = int(np.round((dt_end - dt_start).total_seconds()/dt))*nbaselines
	if no_intervals == 0:
		no_intervals = nbaselines
	# 	j = i + no_intervals
	# else:
	# 	j = i + 666




	times = tb.getcol("TIME",i,no_intervals).reshape(-1, nbaselines)
	uvws = tb.getcol("UVW",i,no_intervals).reshape(3,-1, nbaselines)
	data = tb.getcol("DATA",i,no_intervals)[:,0,:].reshape(4,-1, nbaselines)
	vis = tb.getcol("CORRECTED_DATA",i,no_intervals)[:,0,:].reshape(4,-1, nbaselines) #("CORR_NO_BEAM",i,no_intervals)
	weights = tb.getcol("WEIGHT_SPECTRUM",i,no_intervals)[:,0,:].reshape(4,-1, nbaselines)
	#mdl = tb.getcol("MODEL_DATA",i,no_intervals).reshape(4,-1, nbaselines)
	ant0 = tb.getcol("ANTENNA1",i,no_intervals).reshape(-1, nbaselines)[0]
	ant1 = tb.getcol("ANTENNA2",i,no_intervals).reshape(-1, nbaselines)[0]

	tb.close()
	tb.open(SB+"SPECTRAL_WINDOW")
	freq = tb.getcol("CHAN_FREQ")[0,0]
	sb_width = tb.getcol("CHAN_WIDTH")[0,0]

	tb.close()


	#get data for specified time range
	# i=0
	# j=0
	# for t in times:
	#     dt = epoch_start + timedelta(seconds=t)
	#     if dt < dt_start:
	#         i+=1
	#     if dt < dt_end:
	#         j+=1

	# times = np.mean(times, axis=0)
	# uvws = np.mean(uvws,axis=1)
	# data = np.sum(data, axis=1)
	# vis = np.sum(vis, axis=1)
	# weights = np.sum(weights, axis=1)
	# mdl = np.sum(mdl, axis=1)
	outfile = SB.split("_")[-2]+"MS_data" #"quiet1"




	# cal_MS = "L401001_SB320_uv.dppp.MS/instrument"
	# tb.open(cal_MS)

	# cals = tb.getcol('VALUES')[0]

	# G00r = cals[:,0::4]
	# G00i = cals[:,1::4]
	# G11r = cals[:,2::4]
	# G11i = cals[:,3::4]

	# for G in [G00r, G00i, G11r, G11i]:
	# 	G[np.where(np.isnan(G))]=0

	# abs_sq_g0 = G00r**2 + G00i**2
	# abs_sq_g1 = G11r**2 + G11i**2

	# # b = abs_sq_g0 + abs_sq_g1
	# # c = abs_sq_g0 * abs_sq_g1

	# eig_max = np.max([abs_sq_g0,abs_sq_g1],axis=0)#(b+np.sqrt(b**2-4*c))/2  #Max eigenvalue of Gain matrix = ||G||^2
	# eig_max = np.mean(eig_max, axis=0)




	#tb.close()

	return outfile, freq, times, dt,sb_width,uvws,ant0,ant1,data,vis,weights#,mdl,eig_max


for i in range(76,77):
	SB = "L401003_SB{}_uv.dppp.MS/".format(str(int(i)).zfill(3))
	t_start ="2015-10-17T13:21:40.000"#"2015-10-17T13:21:53.900" #"2015-10-17T12:00:00"#"2015-10-17T13:21:20" #"2015-10-17T13:21:53"
	t_end = "2015-10-17T13:22:00.000"#"2015-10-17T13:21:54.000"#"2015-10-17T12:00:05"#"2015-10-17T13:22:00"#"2015-10-17T13:23:00"
	print("Saving for " + SB)

	outfile, freq, times, dt,sb_width,uvws,ant0,ant1,data,vis,weights = get_MS_data(SB, t_start,t_end)
	np.savez(outfile, freq=freq, times=times, dt=dt, df=sb_width, uvws=uvws, ant0=ant0, ant1=ant1, data=data,vis=vis, weights=weights)









