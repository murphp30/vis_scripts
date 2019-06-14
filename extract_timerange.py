#!/usr/bin/env python2

from pyrap.measures import measures
import pyrap.tables as tables
import os
import sys
import glob


dm = measures()

sas_id = sys.argv[1]
start_date_utc = sys.argv[2]
duration_s = float(sys.argv[3])

input_dir = '/data/'+sas_id+'/'
input_msses = [msname for msname in sorted(glob.glob(input_dir+'*_uv.MS'))
               if msname.split('_')[2][-1] == '0' and msname.split('_')[2][0:2] == 'SB']

start_mjds = dm.epoch('UTC', start_date_utc)['m0']['value']*24*3600.0
end_mjds = start_mjds + duration_s

query = '%.2f <= TIME  AND TIME <= %.2f' % (start_mjds, end_mjds)


output_dir = '/globaldata/scratch/solar-eclipse/'+start_date_utc.split()[-1]+'/'

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for msname in input_msses:
    msout = output_dir+os.path.basename(msname)[:-3]+'_casa.MS'
    print msname+' -> '+msout+' from '+start_date_utc+' for '+str(duration_s)+' seconds.'
    selection = tables.table(msname).query(query)
    selection.copy(msout, deep=True, valuecopy=True)
print 'Done'
