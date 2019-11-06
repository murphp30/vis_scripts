#! /usr/bin/env python
import numpy as np
import lofar.parmdb as pb
from argparse import ArgumentParser
import sys
import os
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pdb
parser = ArgumentParser("Simple parm filter")
parser.add_argument('parmdb',help='input parmdb',metavar='parmdb')
#parser.add_argument('-o','--out',help='output parmdb')

# Global variables
epoch_start = datetime(1858,11,17)
burst_start = datetime(2015,10,17,13,21,43)
burst_end = datetime(2015,10,17,13,22,12) 

def filter_values(values):
    return np.ones_like(values)*np.average(values)
def change_format_of_array(array):    
    replace=[]
    for jj in array:
        replace.append([jj])
    print('same format to put into table',len(replace), np.array(replace))
    return(replace)
def make_rows_cols(Num):
    rows,columns=[],[]
    for i in range(Num):
        n=np.full((1,Num),i)[0]
        rows.append(n)
        n=np.arange(0,Num,1)
        columns.append(n)
    columns=np.concatenate(columns)
    rows=np.concatenate(rows)
    return(rows,columns)
def smoothing(time,array):
    #remove section between 19650 to 19800 # fo
    st,end=19650,19800  # time in seconds where burst starts and ends - find out using parmdbplot.py
    time_res=0.25034770514603616
  #  plt.figure()
  #  plt.plot(time,array,color='red',alpha=.5) #original
    region=300
    x_new=time[st-region:end+region]
   
    ys,xs=[],[]
    ys.append((array[st-region:st]))
    ys.append((array[end:end+region]))
    ys=np.concatenate(ys)
    
    xs.append(time[st-region:st])
    xs.append(time[end:end+region])
    xs=np.concatenate(xs)
   # f = interp1d(xs, ys, kind='cubic')
    z = np.polyfit(xs, ys, 50)
    f = np.poly1d(z)
    array[st-region:end+region]=f(x_new)
  #  new_array=array
 #   plt.plot(time,array,'.',ms=1,color='blue')   
  #  plt.show()
    replace=change_format_of_array(array)    
    return(np.array(replace),array)
def make_subplot(iparm,parm,time,or_array,fixed_array,axsp):
        axsp.plot(time,or_array,'-',color='red',linewidth=1.3,alpha=.2) # original
        axsp.plot(time,fixed_array,'.',ms=1,color='black',alpha=.2) # fixed
        axsp.plot(time,fixed_array,'-',ms=1,color='black',linewidth=.2) # fixed
        axsp.set_title(parm,fontsize=6)
       # axsp.set_xlim(19e3,20e3)
        axsp.set_ylim(-250,250)

def burst_time_index(times, burst_start=burst_start, burst_end=burst_end):
	# Calculate indices where the burst occurs when 
	# given a datetime for burst_start and burst_end
	times = np.array([epoch_start + timedelta(seconds=times[t]) for t in range(len(times))])
	ind_start = np.where(times >= burst_start)[0][0]
	ind_end = np.where(times <= burst_end)[0][-1]
	return ind_start, ind_end

def fix_burst(times, data):
	# Fit a a degree 3 polynomial to before the burst
	# Replace data after burst start with polynomial
	# +/- std of pre burst data (will get to this later).
	# We don't actually care about the data after
	# the burst so it doesn't matter that they don't
	# match up
	# I should have just commented with inverted commas
	
	b_start, b_end = burst_time_index(times)
	fixed_data = np.zeros(data.shape)
	fixed_data[:b_start] = data[:b_start]
	fixed_data[b_end:] = data[b_end:]
	ts = np.arange(len(data))
	m = (data[b_start]-data[b_end])/(ts[b_start]-ts[b_end])
	c = data[b_start]- m*ts[b_start]
	fixed_data[b_start:b_end] = m*ts[b_start:b_end].reshape(data[b_start:b_end].shape) + c
	pre_fit = 10 #fit data to pre_fit seconds before the burst
	poly = np.polyfit(ts[b_start-pre_fit:b_end+pre_fit],fixed_data[b_start-pre_fit:b_end+pre_fit],3)
	p_fit = poly[0]*ts**3 + poly[1]*ts**2 + poly[2]*ts + poly[3]
	p_fit = p_fit.reshape(fixed_data.shape)
	fixed_data[b_start:b_end] = p_fit[b_start:b_end]
	#fixed_data[b_start:] = p_fit#[b_start:]
	return fixed_data

#def main(argv):
argv = sys.argv[1:]
print('---------------------')
print('loading table')
print(argv)
print('---------------------')

args=parser.parse_args(argv)
# args=parser.parse_args(['L401011_SB403_uv.dppp.MS/instrument/'])
newpb=pb.parmdb(args.parmdb+"NEW",create=True)
# pdb.set_trace()
newparms={} #dictionary with a copy of eveerything of the old parms, but new filtered values
parms=pb.parmdb(args.parmdb).getValuesGrid("*")
Nparms = len(parms)
Num = int(np.ceil(np.sqrt(Nparms))) # number of plots needed - each station has phase and amp - then each pahse and amp has real and imag
# fp, axp = plt.subplots(Num, Num,sharex=True, sharey=True, figsize=(20,20))
#make the subplot notation correct
# rows,columns=make_rows_cols(Num)
# count=1
# pdb.set_trace()
for parm in sorted(parms.keys()):
    print('---------------------')
    print('resolving corrupted calibration data')
    print(parm)
    print('---------------------')
    newparm=parms[parm].copy()
    # print(parms[parm])       
    val = parms[parm]['values']
    times = parms[parm]['times']
    # print('old',len(val), val)
    # original_array=np.concatenate(val)
    # time = np.linspace(0,len(original_array), len(original_array))
#   plt.figure()
#    plt.plot(val,'.')
   # newparm['values'],fixed_array = smoothing(time,original_array)
    
    if str(parm)[14:16]=='CS':
        # print()           
        newparm['values'] = fix_burst(times, val) #smoothing(time,original_array)
    # else:
    #     newparm['values']=parms[parm]['values']
    #     fixed_array=original_array
    # make_subplot(count,parm,time,original_array,fixed_array,axp[rows[count],columns[count]])
    
   # plt.plot(fixed_array,'--',alpha=.5)
#    plt.xlim(19e3,20e3)
#    plt.show()
    # count=count+1
    newparms[parm] = newparm #update new array
# fp.subplots_adjust(wspace=0.8, hspace=0.8)  
    #fp.savefig("fixed_solutions.png",dpi=500) 
newpb.addValues(newparms) #save to tables
 
# if __name__ == '__main__':
#     main(sys.argv[1:])    