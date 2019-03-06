#!/bin/bash

for i in $(seq -w 50 50 500);
do
	mkdir -p /mnt/murphp30_data/typeIII_int/SB064/iters_${i}
	cd /mnt/murphp30_data/typeIII_int/SB064/iters_${i}
	wsclean -j 36 -no-reorder -no-update-model-required -weight uniform -size 1024 1024 -scale 10.5469asec -pol I -data-column CORRECTED_DATA -niter $i -intervals-out 300 -interval 77000 77300 -fit-beam -make-psf /mnt/murphp30_data/typeIII_int/L401003_SB064_uv.dppp.MS
	
	mkdir images
	mv *image.fits images
	cd -
done

