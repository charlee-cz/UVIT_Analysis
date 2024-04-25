Assorted python scripts to handle the various stages of background estimation, source detection, aperture photometry and results tabulation for UVIT data.

uvit_fields_headers.py: creates a table of organized field IDs, and general target/exposure time info from the intial fits images. Also creates footprints.pdf, an overview of the NGVS, UVIT, and GALEX FUV coverage of the Virgo Cluster.

get_threshholds.py: replicates the procedure from Mondal et al. (2023) to determine the best choice of sigma threshhold above the background during source detection.

make_backgrounds.py and make_backgrounds_m87.py: they...make backgrounds. As well as rms and segmentation maps, and final background-subtracted science images. There is a separate script to handle the slightly different format of the M87 exposures, but all output is created to be consistent with the remainder of the UVIT fields so that any subsequent analysis can be done all together.

get_photometry: contains all code to extract magnitudes from the UVIT fields, using both GALEX-derived apertures and optical NGVS apertures. Creates files containing measurements for each UVIT field, as well as reference figures showing the apertures on the sky, and (for GALEX) the magnitude differences between those measured in the GALEX imaging and those from the new UVIT data.
