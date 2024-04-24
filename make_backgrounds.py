#!/Users/sargas/miniconda3/bin/python

'''
**************************************************************************************************
**** HEY IF YOU RUN THIS THERE ARE LOTS OF HARDCODED FILE LOCATIONS IN HERE TO CHANGE!! sorry.****
**************************************************************************************************

All-in-one script for iterative generation of backgrounds and segmentation maps!

Before background estimation and source detection, the input images are smoothed with a Gaussian kernel, with the FWHM set to the typical FUV seeing (1.5").

First we estimate a very coarse estimate of the background with an 2400 x 2400 pixel mesh (the enormous mesh is to ensure the mean background is dominated by background pixels and not (extended) sources. The high number of zero-value pixels means that the typical median background value is not helpful (spoiler: it's zero).

Then a first past at source detection is completed, selecting all pixels 0.5 sig above the initial background estimate. This is pretty generous since at this point the estimated background level is most likely over-estimated so there is less distinction between the background and sources. Some adjustments to the chosen sigma value were made for the <500sec exposures and other cases that weren't working out well.

All flagged source pixels plus a 10 pix buffer around them are masked while a second background is estimated -- this time with a 800 x 800 mesh. Once again, we run source detection with the new background estimate and same sigma limit and create a revised source mask.

Finally we create the last background map with a 300 x 300 pixel mesh and the latest source mask -- at this point, there should be minimal
source contamination in the background levels (hopefully). We do one last pass of source detection (plus deblending) to create the final segmentation map, this time using the best sigma for each exposure time determined with the tests in FILENAME.py.

The above process is tailored to get background and RMS maps unaffected by extended sources (galaixes). But for point sources that can often be very close to or on top of extended sources, ideally we'd like RMS maps that factor in the light from those extended sources and the smaller-scale changes in the background level. So this code also creates a set of "PSF segmaps" with a 32x32 mesh and no source masking. It creates a segmap based on anything 1 sigma above the background level.

This script produces several figures and fits files:

- *_masked_bkg2.pdf: the UVIT field, with sources masked in black, and the remaining background pixels smoothed and displayed in greyscale. Mainly a diagnostic to check if any bright source pixels are not being picked up in the seg maps.
- *_bkg_new.fits: the final background map for each UVIT field
- *_bkg_rms_new.fits: the final RMS map for each UVIT field
- *_segmap_new.fits: the final deblended segmentation map for each UVIT field

There are also versions of each of the above created using the PSF-suited parameters. 
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

pada = int((4800 - 2.0*pixels)/2.0)
padb = pada #- 1

foot = np.pad(mask, [(pada, padb), (pada, padb)], mode='constant', constant_values=0)
coverage_mask = (foot == 0)

bkg_estimator = MeanBackground(sigma_clip=None)

# Set up kernal to smooth image with Gaussian FWHM = 3.5 pixels (1.5") before detecting sources
sigma = 3.5 * gaussian_fwhm_to_sigma  
kernel = Gaussian2DKernel(sigma)
kernel.normalize()

bestthreshes = [0.6, 0.5, 0.8, 1.0, 1.25]
exps = np.array([500, 4000, 7500, 11000, 20000])

for fullpath in glob.iglob('/Users/sargas/Documents/UVIT/A*/*.fits', recursive=True):
	filename = os.path.basename(fullpath)
	if 'EXPARRAY' not in filename:
		if 'BaF2' in filename:
			image = fits.open(fullpath)

			bestthresh = bestthreshes[np.argmin(abs(image[0].header['RDCDTIME'] - exps))]
			print(bestthresh, image[0].header['RDCDTIME'])
			print('Procesesing %s...' % filename)

			data = image[0].data / image[0].header['RDCDTIME']*foot
			imwcs = wcs.WCS(image[0].header, image)
			obj = image[0].header['OBJECT']

			if obj in ['VCC1588', 'VCC1524']:
				bestthresh = 0.8

			convolved_data = convolve(data, kernel)

			bkg = Background2D(data, (2400,2400), filter_size=(5, 5), coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=75.0)
			print(bkg.background_median, bkg.background_rms_median)

			if image[0].header['RDCDTIME'] < 500.:
				threshold = bkg.background + (bestthresh * bkg.background_rms)
			else:
				threshold = bkg.background + (0.5 * bkg.background_rms)

			segm_detect = detect_sources(convolved_data, threshold, npixels=10)
			footprint = circular_footprint(radius=10)
			mask = segm_detect.make_source_mask(footprint=footprint)

			maskedblur = np.where(mask == 0, convolved_data, 0)
			new_mask = np.where(mask == 0, 0, 1)

			bkg = Background2D(data, (800,800), filter_size=(5, 5), mask=new_mask, coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=75.0)
			# print(bkg.background_median, bkg.background_rms_median)

			if image[0].header['RDCDTIME'] < 500.:
				threshold = bkg.background + (bestthresh * bkg.background_rms)
			else:
				threshold = bkg.background + (0.5 * bkg.background_rms)

			segm_detect = detect_sources(convolved_data, threshold, npixels=10)
			footprint = circular_footprint(radius=10)
			mask = segm_detect.make_source_mask(footprint=footprint)
			new_mask = np.where(mask == 0, 0, 1)

			bkg = Background2D(data, (300,300), filter_size=(5, 5), mask=new_mask, coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=75.0)
			# print(bkg.background_median, bkg.background_rms_median)

			threshold = bkg.background + (bestthresh * bkg.background_rms)

			segm_detect = detect_sources(convolved_data, threshold, npixels=10)
			segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)

			maskedblur = np.where(segm_deblend.data == 0, convolved_data, 0)
			fig = plt.figure(figsize=(8, 8))
			ax = plt.subplot(projection=imwcs)
			plt.imshow(maskedblur, origin='lower', cmap='gist_gray', norm=simple_norm(maskedblur, 'sqrt', percent=99.))
			plt.savefig('Diagnostics/%s_masked_bkg.pdf' % fieldmatch[obj])
			plt.close(fig)

			# Save background map and rms image
			bkg_hdu = fits.PrimaryHDU(bkg.background, header=imwcs.to_header())
			rms_hdu = fits.PrimaryHDU(bkg.background_rms, header=imwcs.to_header())

			# save science image
			sci_img = data - bkg.background
			sci_hdu = fits.PrimaryHDU(sci_img, header=image[0].header)
			sci_hdu.writeto('science_imgs/%s_sci.fits' % fieldmatch[obj], overwrite=True)

			# Save background map and rms image
			rms_hdu.writeto('science_imgs/%s_rms.fits' % fieldmatch[obj], overwrite=True)
			bkg_hdu.writeto('science_imgs/%s_bkg.fits' % fieldmatch[obj], overwrite=True)

			# Save segmentation map of detected objects
			segm_hdu = fits.PrimaryHDU(segm_deblend.data.astype(np.uint32), header=imwcs.to_header())
			segm_hdu.writeto('science_imgs/%s_segmap.fits' % fieldmatch[obj], overwrite=True)

			bkg_psf = Background2D(data, (32,32), filter_size=(5, 5), coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
			# print(bkg_psf.background_median, bkg_psf.background_rms_median)

			threshold = bkg_psf.background + (1.0 * bkg_psf.background_rms)

			segm_detect = detect_sources(convolved_data, threshold, npixels=10)
			segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)

			maskedblur = np.where(segm_deblend.data == 0, convolved_data, 0)
			fig = plt.figure(figsize=(8, 8))
			ax = plt.subplot(projection=imwcs)
			plt.imshow(maskedblur, origin='lower', cmap='gist_gray', norm=simple_norm(maskedblur, 'sqrt', percent=99.))
			plt.savefig('Diagnostics/%s_masked_bkg_psf.pdf' % fieldmatch[obj])
			plt.close(fig)

			# Save background map and rms image
			bkg_hdu = fits.PrimaryHDU(bkg_psf.background, header=imwcs.to_header())
			rms_hdu = fits.PrimaryHDU(bkg_psf.background_rms, header=imwcs.to_header())

			bkg_hdu.writeto('science_imgs/%s_bkg_psf.fits' % fieldmatch[obj], overwrite=True)
			rms_hdu.writeto('science_imgs/%s_rms_psf.fits' % fieldmatch[obj], overwrite=True)

			# Save segmentation map of detected objects
			segm_hdu = fits.PrimaryHDU(segm_deblend.data.astype(np.uint32), header=imwcs.to_header())
			segm_hdu.writeto('science_imgs/%s_segmap_psf.fits' % fieldmatch[obj], overwrite=True)



