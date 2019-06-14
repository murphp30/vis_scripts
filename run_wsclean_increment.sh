#!/bin/bash

for i in 000010 000050 000100 000500 001000 005000 010000 050000 100000 500000;
do
		mkdir -p /mnt/murphp30_data/typeIII_int/iter_test/burst/iter_${i}
		mkdir -p /mnt/murphp30_data/typeIII_int/iter_test/no_burst/iter_${i}
		cd /mnt/murphp30_data/typeIII_int/iter_test/burst/iter_${i}
		#msoverview in=../L401003_SB${i}_uv.dppp.MS > ./ms_txt.txt
		#clean_scale=$(/mnt/murphp30_data/typeIII_int/scripts/SB_freq.py ./ms_txt.txt)
		wsclean -j 36 -mem 85 -no-reorder -no-update-model-required -weight briggs -1 -mgain 0.85 -size 2048 2048 -scale 2.8752asec -pol I -data-column CORRECTED_DATA -niter ${i} -intervals-out 1 -interval 77194 77195 -use-differential-lofar-beam -fit-beam -make-psf /mnt/murphp30_data/typeIII_int/L401003_SB200_uv.dppp.MS
		mv wsclean-image.fits ../wsclean-iter${i}-image.fits
		
		cd /mnt/murphp30_data/typeIII_int/iter_test/no_burst/iter_${i}
		wsclean -j 36 -mem 85 -no-reorder -no-update-model-required -weight briggs -1 -mgain 0.85 -size 2048 2048 -scale 2.8752asec -pol I -data-column CORRECTED_DATA -niter ${i} -intervals-out 1 -interval 77265 77266 -use-differential-lofar-beam -fit-beam -make-psf /mnt/murphp30_data/typeIII_int/L401003_SB200_uv.dppp.MS
		#mkdir images_samescale
		mv wsclean-image.fits ../wsclean-iter${i}-image.fits
done

