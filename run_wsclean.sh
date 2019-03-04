#!/bin/bash

for i in $(seq -w 0 50 243);
do
	for b in -2 -1 0 1 2;
	do
		mkdir -p /mnt/murphp30_data/typeIII_int/SB${i}/Briggs_${b}
		cd /mnt/murphp30_data/typeIII_int/SB${i}/Briggs_${b}
		wsclean -j 36 -no-reorder -no-update-model-required -weight briggs $b -size 1024 1024 -scale 10.5469asec -pol I -data-column CORRECTED_DATA -niter 40 -intervals-out 300 -interval 77000 77300 -fit-beam -make-psf /mnt/murphp30_data/typeIII_int/L401003_SB${i}_uv.dppp.MS
		
#		sleep 900
		mkdir images
		mv *image.fits images
		cd -
	done
done

