#!/bin/bash

for i in $(seq -w 50 50 243);
do
		mkdir -p /mnt/murphp30_data/typeIII_int/SB${i}
		cd /mnt/murphp30_data/typeIII_int/SB${i}
		#msoverview in=../L401003_SB${i}_uv.dppp.MS > ./ms_txt.txt
		#clean_scale=$(/mnt/murphp30_data/typeIII_int/scripts/SB_freq.py ./ms_txt.txt)
		wsclean -j 36 -mem 85 -no-reorder -no-update-model-required -weight briggs -1 -mgain 0.85 -size 2048 2048 -scale 2.8752asec -pol I -data-column CORRECTED_DATA -niter 40 -intervals-out 300 -interval 77000 77300 -use-differential-lofar-beam -fit-beam -make-psf /mnt/murphp30_data/typeIII_int/L401003_SB${i}_uv.dppp.MS
		
		mkdir images_samescale
		mv *image.fits images_samescale
		cd -
done

