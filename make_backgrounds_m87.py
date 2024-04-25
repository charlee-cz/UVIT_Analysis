#!/Users/sargas/miniconda3/bin/python

'''
**************************************************************************************************
**** HEY IF YOU RUN THIS THERE ARE LOTS OF HARDCODED FILE LOCATIONS IN HERE TO CHANGE!! sorry.****
**************************************************************************************************

All-in-one script for iterative generation of backgrounds and segmentation maps!

This code does the exact same thing as make_backgrounds.py, except there have been slight edits to the mesh sizes, header keywords, and other initial formatting to account for the slightly different setup of the M87 images. The fits images created by this script will match the image dimensions of the others, though.
'''

import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
from astropy.table import QTable
from astropy.convolution import convolve
import astropy.units as u
from astropy.visualization import simple_norm
import astropy.wcs as wcs
import astroimtools

from photutils.background import Background2D, MeanBackground, SExtractorBackground
from photutils.segmentation import detect_sources, deblend_sources
from photutils.utils import circular_footprint

import matplotlib.pyplot as plt
import glob, os

field, vccs = np.genfromtxt('uvit_header_info.dat', usecols=(0, 4), dtype=str, unpack=True)
fieldmatch = {}
for i,vcc in enumerate(vccs):
	fieldmatch[vcc] = field[i]

pixels = 14*60./0.417 # number of pixels in 14' given pixel scale
mask = astroimtools.circular_footprint(pixels)

pada = int((4096 - 2.0*pixels)/2.0)
padb = pada #- 1

foot = np.pad(mask, [(pada, padb), (pada, padb)], mode='constant', constant_values=0)

# additional padding to make M87 images match the other UVIT image dimensions
newpad = int((4800 - 4096)/2)
newfoot = np.pad(foot, (newpad, newpad), mode='constant', constant_values=0)
coverage_mask = (newfoot == 0)

bkg_estimator = MeanBackground(sigma_clip=None)

# Set up kernal to smooth image with Gaussian FWHM = 3.5 pixels (1.5") before detecting sources
sigma = 3.5 * gaussian_fwhm_to_sigma  
kernel = Gaussian2DKernel(sigma)#, x_size=3, y_size=3)
kernel.normalize()

bestthreshes = [0.6, 0.5, 0.8, 1.0, 1.25]
exps = np.array([500, 4000, 7500, 11000, 20000])

for fullpath in glob.iglob('/Users/sargas/Documents/UVIT/M87/*MASTER.fits', recursive=True):
	filename = os.path.basename(fullpath)
	image = fits.open(fullpath)

	bestthresh = bestthreshes[np.argmin(abs(image[0].header['TOTITIME'] - exps))]
	print('Procesessing %s...' % filename)

	data = image[0].data * foot
	data = np.pad(data, (newpad, newpad), mode='constant', constant_values=0)
	image[0].header['CRPIX1'] = image[0].header['CRPIX1'] + newpad
	image[0].header['CRPIX2'] = image[0].header['CRPIX2'] + newpad
	imwcs = wcs.WCS(image[0].header, image)
	obj = image[0].header['OBJECT']
	filt = image[0].header['DETECTOR']

	if filt == 'FUV':
		bestthresh = 0.75
	else:
		bestthresh = 1.2

	convolved_data = convolve(data, kernel)
	
	bkg = Background2D(data, (2400,2400), filter_size=(5, 5), coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
	print(bkg.background_median, bkg.background_rms_median)

	threshold = bkg.background + (bestthresh * bkg.background_rms)

	segm_detect = detect_sources(convolved_data, threshold, npixels=10)
	footprint = circular_footprint(radius=10)
	mask = segm_detect.make_source_mask(footprint=footprint)

	maskedblur = np.where(mask == 0, convolved_data, 0)
	new_mask = np.where(mask == 0, 0, 1)

	bkg = Background2D(data, (800,800), filter_size=(5, 5), mask=new_mask, coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
	# print(bkg.background_median, bkg.background_rms_median)

	threshold = bkg.background + (bestthresh * bkg.background_rms)

	segm_detect = detect_sources(convolved_data, threshold, npixels=10)
	footprint = circular_footprint(radius=10)
	mask = segm_detect.make_source_mask(footprint=footprint)
	new_mask = np.where(mask == 0, 0, 1)

	bkg = Background2D(data, (300,300), filter_size=(5, 5), mask=new_mask, coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
	# print(bkg.background_median, bkg.background_rms_median)

	if filt == 'FUV':
		threshold = bkg.background + (1.0 * bkg.background_rms)
	else:
		threshold = bkg.background + (2.0 * bkg.background_rms)

	segm_detect = detect_sources(convolved_data, threshold, npixels=10)
	segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)

	maskedblur = np.where(segm_deblend.data == 0, convolved_data, 0)
	fig = plt.figure(figsize=(8, 8))
	ax = plt.subplot(projection=imwcs)
	plt.imshow(maskedblur, origin='lower', cmap='gist_gray', norm=simple_norm(data, 'sqrt', percent=99.))
	if filt=='NUV':
		plt.savefig('Diagnostics/%s_%s_masked_bkg.pdf' % (fieldmatch['VCC1316'], filt))
	else:
		plt.savefig('Diagnostics/%s_masked_bkg.pdf' % fieldmatch['VCC1316'])
	plt.close(fig)

	sci_img = data - bkg.background
	sci_hdu = fits.PrimaryHDU(sci_img, header=image[0].header)
	if filt=='NUV':
		sci_hdu.writeto('science_imgs/%s_%s_sci.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
	else:
		sci_hdu.writeto('science_imgs/%s_sci.fits' % fieldmatch['VCC1316'], overwrite=True)

	# Save background map, rms image and segmentation map
	bkg_hdu = fits.PrimaryHDU(bkg.background, header=imwcs.to_header())
	rms_hdu = fits.PrimaryHDU(bkg.background_rms, header=imwcs.to_header())
	segm_hdu = fits.PrimaryHDU(segm_deblend.data.astype(np.uint32), header=imwcs.to_header())

	if filt=='NUV':
		bkg_hdu.writeto('science_imgs/%s_%s_bkg.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
		rms_hdu.writeto('science_imgs/%s_%s_rms.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
		segm_hdu.writeto('science_imgs/%s_%s_segmap.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
	else:
		bkg_hdu.writeto('science_imgs/%s_bkg.fits' % fieldmatch['VCC1316'], overwrite=True)
		rms_hdu.writeto('science_imgs/%s_rms.fits' % fieldmatch['VCC1316'], overwrite=True)
		segm_hdu.writeto('science_imgs/%s_segmap.fits' % fieldmatch['VCC1316'], overwrite=True)
	
	# Now make a background map tailored for PSF sources that includes background from extended sources
	bkg_psf = Background2D(data, (32,32), filter_size=(5, 5), coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
	# print(bkg_psf.background_median, bkg_psf.background_rms_median)

	if filt == 'FUV':
		threshold = bkg_psf.background + (1.0 * bkg_psf.background_rms)
	else:
		threshold = bkg_psf.background + (2.0 * bkg_psf.background_rms)

	segm_detect = detect_sources(convolved_data, threshold, npixels=10)
	segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)

	maskedblur = np.where(segm_deblend.data == 0, convolved_data, 0)
	fig = plt.figure(figsize=(8, 8))
	ax = plt.subplot(projection=imwcs)
	plt.imshow(maskedblur, origin='lower', cmap='gist_gray', norm=simple_norm(maskedblur, 'sqrt', percent=99.))
	if filt=='NUV':
		plt.savefig('Diagnostics/%s_%s_masked_bkg_psf.pdf' % (fieldmatch['VCC1316'], filt))
	else:
		plt.savefig('Diagnostics/%s_masked_bkg_psf.pdf' % fieldmatch['VCC1316'])
	plt.close(fig)

	# Save background map, rms image and segmentation map
	bkg_hdu = fits.PrimaryHDU(bkg_psf.background, header=imwcs.to_header())
	rms_hdu = fits.PrimaryHDU(bkg_psf.background_rms, header=imwcs.to_header())
	segm_hdu = fits.PrimaryHDU(segm_deblend.data.astype(np.uint32), header=imwcs.to_header())

	if filt=='NUV':
		bkg_hdu.writeto('science_imgs/%s_%s_bkg_psf.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
		rms_hdu.writeto('science_imgs/%s_%s_rms_psf.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
		segm_hdu.writeto('science_imgs/%s_%s_segmap_psf.fits' % (fieldmatch['VCC1316'], filt), overwrite=True)
	else:
		bkg_hdu.writeto('science_imgs/%s_bkg_psf.fits' % fieldmatch['VCC1316'], overwrite=True)
		rms_hdu.writeto('science_imgs/%s_rms_psf.fits' % fieldmatch['VCC1316'], overwrite=True)
		segm_hdu.writeto('science_imgs/%s_segmap_psf.fits' % fieldmatch['VCC1316'], overwrite=True)
