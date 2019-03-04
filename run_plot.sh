#!/bin/bash

for i in $(seq -w 0 50 243);
do
#	for b in -2 -1 0 1 2;
	#do
#		python pearse_plot_fits.py -indir /mnt/murphp30_data/typeIII_int/SB${i}/Briggs_${b}/images/
		
	python pearse_plot_fits.py -indir /mnt/murphp30_data/typeIII_int/SB${i}/uniform/images/
#		sleep 900
	#done
done

