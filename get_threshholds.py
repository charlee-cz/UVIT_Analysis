#!/Users/sargas/miniconda3/bin/python

'''
This code uses 5 "representative" images of different exposure times to recreate the method from Mondal+23, which they use to determine the best sigma threshhold for source detection

This code determines the best sigma (based on minimizing the skew of the background level estimated using non-source pixels throughout the image) which will then be used in the official background estimation and source detection in make_backgrounds.py.
'''

import os
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
from astropy.table import QTable
from astropy.convolution import convolve
import astropy.units as u
from astropy.visualization import make_lupton_rgb, SqrtStretch, ImageNormalize, simple_norm
import astropy.wcs as wcs
import astroimtools
from astropy.coordinates import angular_separation, Angle
import astropy
import pandas
import random
import math

import photutils
from photutils.background import Background2D, MeanBackground, SExtractorBackground
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog#, source_properties (new API)
from photutils.utils import calc_total_error

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches
from scipy.interpolate import griddata
import glob, os

#Make a circular mask which excludes everything outside the Astrosat UVIT field of view
pixels = 14*60./0.417 # number of pixels in 14' given pixel scale
mask = astroimtools.circular_footprint(pixels)

pada = int((4800 - 2.0*pixels)/2.0)
padb = pada #- 1

foot = np.pad(mask, [(pada, padb), (pada, padb)], mode='constant', constant_values=0)
coverage_mask = (foot == 0)

bkg_estimator = MeanBackground(sigma_clip=None)

# Set up kernal to smooth image with Gaussian FWHM = 3.5 pixels (1.5") before detecting sources
sigma = 3.5 * gaussian_fwhm_to_sigma  
kernel = Gaussian2DKernel(sigma) #, x_size=3, y_size=3)
kernel.normalize()

exps = [300, 4000, 7500, 11000, 20000]

# '/Users/sargas/Documents/UVIT/A10_071/VCC1588_FUV_BaF2___MASTER_IMAGE_A10_071T66.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC1078_FUV_BaF2___MASTER_IMAGE_A10_071T54.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC792_FUV_BaF2___MASTER_IMAGE_A10_071T12.fits', 
for j,fullpath in enumerate(['/Users/sargas/Documents/UVIT/A10_071/VCC491_FUV_BaF2___MASTER_IMAGE_A10_071T08.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC1588_FUV_BaF2___MASTER_IMAGE_A10_071T66.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC1078_FUV_BaF2___MASTER_IMAGE_A10_071T54.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC792_FUV_BaF2___MASTER_IMAGE_A10_071T12.fits', '/Users/sargas/Documents/UVIT/A10_071/VCC89_FUV_BaF2___MASTER_EXPARRAY_A10_071T01.fits']):
	image = fits.open(fullpath)


	#Rescale to counts per second and zero out anything outside footprint
	data = image[0].data * foot / image[0].header['RDCDTIME']
	#data = image[0].data * foot
	filt = image[0].header['FILTERID']
	obj = image[0].header['OBJECT']
	imwcs = wcs.WCS(image[0].header, image)

	#Rescale pixel area and check dimensions of image
	ny, nx = data.shape
	pixscale = wcs.utils.proj_plane_pixel_scales(imwcs)[0] 
	pixscale *= imwcs.wcs.cunit[0].to('arcsec')

	# pretty sure we don't want to use this -- but it was going to be a rudimentary source mask determined before background estimation or source detection
	test_mask = np.where(data > np.percentile(data, 99.7), 1, 0)

	#outline = '%d x %d pixels' % (ny, nx)
	#outline += ' = %g" x %g"' % (ny * pixscale, nx * pixscale)
	#outline += ' (%.2f" / pixel)' % pixscale
	#print(outline)

	# Measure background and set detection threshold
	print('Estimating background...')
	bkg = Background2D(data, (800,800), filter_size=(3, 3), coverage_mask=coverage_mask, sigma_clip=None, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
	#bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)
	print(bkg.background_median, bkg.background_rms_median)

	print('Convolving...')
	convolved_data = convolve(data, kernel)
	
	for backthresh in np.linspace(0.5, 3, num=51):
		print('Processing for threshold=%.2f...' % backthresh)

		# a klugey workaround in case we get a background of entirely zero but I'm 99.9% sure this is no longer an issue now that we're using mean background estimators above (instead of median)
		if (bkg.background_median == bkg.background_rms_median):
			threshold = bkg.background + (backthresh * bkg.background_rms) + (0.2 / image[0].header['RDCDTIME'])
		else:
			threshold = bkg.background + (backthresh * bkg.background_rms) #+ (0.1 / image[0].header['RDCDTIME'])
		
		#fig = plt.figure(figsize=(8, 8))
		#plt.imshow(threshold, origin='lower', cmap='Greys_r', interpolation='nearest', norm=simple_norm(data, 'sqrt', percent=95.))
		#plt.show()
		
		# Detect and deblend #note: changed npixels = 5 to 10
		print('\tDetecting sources...')
		segm_detect = detect_sources(convolved_data, threshold, npixels=10)
		# Detect and deblend #note: changed npixels = 5 to 10
		#print('\tDeblending...')
		#segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)

		#fig = plt.figure(figsize=(8, 8))
		#plt.imshow(segm_detect.data, origin='lower', cmap='Greys_r', interpolation='nearest', vmax=1)
		#plt.show()

		bigx = [2399, 2360, 3282, 3720, 3368, 2442, 1498, 1000, 1431]
		bigy = [2399, 3750, 3386, 2457, 1524, 1012, 1443, 2364, 3289]
		bigdelt = 426
		lildelt = 5

		#fig = plt.figure(figsize=(6, 6))
		#ax = plt.subplot(projection=imwcs)
		#plt.imshow(data, origin='lower', cmap="Greys_r", norm=simple_norm(data, 'sqrt', percent=99.))
		#plt.xlabel('Right Ascension')
		#plt.ylabel('Declination')
		print('\tBeginning random background box generation.')
		#f = open('/Users/sargas/Documents/UVIT/tests/threshold_%d_%.2f.dat' % (exps[j], backthresh), 'w')
		f = open('/Users/sargas/Documents/UVIT/tests/threshold_%d_%.2f.dat' % (exps[j], backthresh), 'w')
		f.write('#BoxID\tx\ty\tBackground\trms\n')
		for i in range(len(bigx)):
			numboxes = 0
			#bkgs = []
			#rmss = []
			#points = np.zeros((400, 2))
			#rect = patches.Rectangle((bigx[i]-431, bigy[i]-431), 863, 863, linewidth=1, edgecolor='b', facecolor='none')
			#ax.add_patch(rect)

			while numboxes < 400:
				lilx = random.randint(bigx[i] - bigdelt, bigx[i] + bigdelt)
				lily = random.randint(bigy[i] - bigdelt, bigy[i] + bigdelt)

				if np.sum(segm_detect.data.astype(np.uint32)[lily-lildelt:lily+lildelt+1, lilx-lildelt:lilx+lildelt+1]) == 0:
					#rect = patches.Rectangle((lilx-5, lily-5), 11, 11, linewidth=1, edgecolor='r', facecolor='none')
					#ax.add_patch(rect)
					vals = data[lily-lildelt:lily+lildelt+1, lilx-lildelt:lilx+lildelt+1]
					#bkgs = np.append(bkgs, np.mean(vals)) #calculate background per pixel
					#rmss = np.append(rmss, np.sqrt(np.sum(vals**2)/121. - np.mean(vals)**2))
					#points[numboxes, 0] = lilx
					#points[numboxes, 1] = lily
					f.write('%d\t%d\t%d\t%e\t%e\n' % (i, lilx, lily, np.mean(vals), np.sqrt(np.sum(vals**2)/121. - np.mean(vals)**2)))
					numboxes+=1
					if numboxes % 100 == 0:
						print('\t\t...still working...')
				else:
					pass
		print('\t...box generation complete.')
		f.close()
	image.close()

exps = [300, 4000, 7500, 11000, 20000]
# Read in all the backgrounds from random boxes and make histograms, check skew
for exp in exps:
	best = 1000.
	for backthresh in np.linspace(0.5, 3, num=51):
		boxnum, stuff = np.genfromtxt('/Users/sargas/Documents/UVIT/tests/threshold_%d_%.2f.dat' % (exp, backthresh), usecols=(0,3), unpack=True)
		skew = []
		for num in range(int(np.max(boxnum))):
			skew = np.append(skew, abs((np.mean(stuff[boxnum == num]) - np.median(stuff[boxnum == num])) / np.std(stuff[boxnum == num])))
		if np.median(skew) < best:
			best = np.median(skew)
			globalbkg = np.mean(stuff)
			globalerr = np.std(stuff)
			bestthresh = backthresh
	print(best, bestthresh)
	print(globalbkg, globalerr)


	boxnum, stuff = np.genfromtxt('/Users/sargas/Documents/UVIT/tests/threshold_%d_1.00.dat' % exp, usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='0.5', linestyle=':', label=r'$1\,\sigma$')
	boxnum, stuff = np.genfromtxt('/Users/sargas/Documents/UVIT/tests/threshold_%d_2.50.dat' % exp, usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='0.5', linestyle='--', label=r'$2.5\,\sigma$')

	boxnum, stuff = np.genfromtxt('/Users/sargas/Documents/UVIT/tests/threshold_%d_%.2f.dat' % (exp, bestthresh), usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='tab:blue', lw=1.5, zorder=10, label=r'$%.2f\,\sigma$' % bestthresh)
	globalbkg = np.mean(stuff)
	globalerr = np.std(stuff)
	print(globalbkg, globalerr, len(stuff))
	plt.legend()
	plt.xlabel(r'Background (counts sec$^{-1}$)')
	plt.ylabel(r'N')
	plt.tight_layout()
	plt.savefig('/Users/sargas/Documents/UVIT/threshold_%d.pdf' % exp)
	plt.clf()


'''
**** everything below is now defunct -- it's an outdated background estimation ****


Create 3600 random boxes with best threshold
Record mean background for each set of boxesRepeat for a total of 100 measurements
Global background is average of those 100 averages
Error in global background is the rms provided by photutils
For one iteration of boxes, check out variation across the field
'''
'''
for fullpath in glob.iglob('/Users/sargas/Documents/UVIT/A*/*.fits', recursive=True):
	filename = os.path.basename(fullpath)
	if 'EXPARRAY' not in filename:
		# ingore sextractor images
		if 'BaF2' in filename:
			image = fits.open(fullpath)

			print('Procesesing %s...' % filename)


			#Rescale to counts per second and zero out anything outside footprint
			data = image[0].data * foot / image[0].header['RDCDTIME']
			#data = image[0].data * foot
			filt = image[0].header['FILTERID']
			obj = image[0].header['OBJECT']
			imwcs = wcs.WCS(image[0].header, image)

			#Rescale pixel area and check dimensions of image
			ny, nx = data.shape
			pixscale = wcs.utils.proj_plane_pixel_scales(imwcs)[0] 
			pixscale *= imwcs.wcs.cunit[0].to('arcsec')

			# For detection, requiring 10 connected pixels 2-sigma above background

			# Measure background and set detection threshold
			bkg = Background2D(data, (300,300), filter_size=(5, 5), coverage_mask=coverage_mask, fill_value=0, bkg_estimator=bkg_estimator, exclude_percentile=50.0)
			#bkg = Background2D(data, (50, 50), filter_size=(3, 3), bkg_estimator=bkg_estimator)

			#print(bkg.background_median, bkg.background_rms_median)
			overall_err = bkg.background_rms_median

			print('Convolving...')
			convolved_data = convolve(data, kernel)
			if overall_err == 0.:
				print('Additional tweaking required for %s.' % obj)
				threshold = bkg.background + (bestthresh * bkg.background_rms) + (0.35 / image[0].header['RDCDTIME'])
				overall_err = np.median(threshold)
			else:
				threshold = bkg.background + ((bestthresh + 0.3) * bkg.background_rms)

				print(bkg.background_median, bkg.background_rms_median)
				#fig = plt.figure(figsize=(8, 8))
				#plt.imshow(bkg.background, origin='lower', cmap='Greys_r', interpolation='nearest')
				#plt.show()


				# Detect and deblend #note: changed npixels = 5 to 10
				print('\tDetecting sources...')
				segm_detect = detect_sources(convolved_data, threshold, npixels=10)
				# Detect and deblend #note: changed npixels = 5 to 10
				print('\tDeblending...')
				#print('%s\t%f' % (obj, image[0].header['RDCDTIME']))
				segm_deblend = deblend_sources(convolved_data, segm_detect, npixels=10, nlevels=32, contrast=0.005)


				# Save segmentation map of detected objects
				segm_hdu = fits.PrimaryHDU(segm_deblend.data.astype(np.uint32), header=imwcs.to_header())
				if 'A08' in filename:
					segm_hdu.writeto('A08_003/UVIT_%s.fits' % obj, overwrite=True)
				else:
					segm_hdu.writeto('A10_071/UVIT_%s.fits' % obj, overwrite=True)

				bigx = [2399, 2360, 3272, 3720, 3358, 2442, 1508, 1050, 1441]
				bigy = [2399, 3750, 3366, 2457, 1514, 992, 1423, 2364, 3269]
				bigdelt = 426
				lildelt = 5

				niters = 0

				fig = plt.figure(figsize=(6, 6))
				ax = plt.subplot(projection=imwcs)
				plt.imshow(data, origin='lower', cmap="Greys", norm=simple_norm(data, 'sqrt', percent=99.))
				circ = patches.Circle((2400, 2400), pixels*13./14., linewidth=1, edgecolor='0.2', linestyle='--', facecolor='none')
				ax.add_patch(circ)
				plt.xlabel('Right Ascension')
				plt.ylabel('Declination')
				print('\tBeginning random background box generation.')
				#f = open('threshold2_%.2f.dat' % backthresh, 'w')
				#f.write('#BoxID\tx\ty\tBackground\trms\n')

				globallbkgs = []
				while niters < 1:
					allbkgs = []
					for i in range(len(bigx)):
						numboxes = 0
						bkgs = []
						#rmss = []
						#points = np.zeros((400, 2))
						if niters == 0:
							rect = patches.Rectangle((bigx[i]-431, bigy[i]-431), 863, 863, linewidth=1, edgecolor='tab:blue', facecolor='none')
							ax.add_patch(rect)

						while numboxes < 400:
							lilx = random.randint(bigx[i] - bigdelt, bigx[i] + bigdelt)
							lily = random.randint(bigy[i] - bigdelt, bigy[i] + bigdelt)

							if np.sum(segm_detect.data.astype(np.uint32)[lily-lildelt:lily+lildelt+1, lilx-lildelt:lilx+lildelt+1]) == 0:
								vals = data[lily-lildelt:lily+lildelt+1, lilx-lildelt:lilx+lildelt+1]
								bkgs = np.append(bkgs, np.mean(vals)) #calculate background per pixel
								allbkgs = np.append(allbkgs, np.mean(vals))
								#rmss = np.append(rmss, np.sqrt(np.sum(vals**2)/121. - np.mean(vals)**2))
								#points[numboxes, 0] = lilx
								#points[numboxes, 1] = lily
								#f.write('%d\t%d\t%d\t%e\t%e\n' % (i, lilx, lily, np.mean(vals), np.sqrt(np.sum(vals**2)/121. - np.mean(vals)**2)))
								numboxes+=1
								if numboxes % 100 == 0:
									print('\t\t...still working...')
							else:
								pass
						if niters==0:
							if i==0:
								# print mean bkg in central box
								# print percentage deviation for other boxes
								plt.text(bigx[i], bigy[i], '%.2e\nmean' % bkgs[0], horizontalalignment='center', verticalalignment='center', color='0.2')
								#rect = patches.Rectangle((lilx-5, lily-5), 11, 11, linewidth=1, edgecolor='r', facecolor='none')
								#ax.add_patch(rect)
							else:
								plt.text(bigx[i], bigy[i], '{:.1f}\%'.format(100. * (bkgs[i] - bkgs[0]) / bkgs[0]), horizontalalignment='center', verticalalignment='center', color='0.2')
					if niters == 0:
						plt.tight_layout()
						if 'A08' in filename:
							plt.savefig('A08_003/UVIT_%s.pdf' % obj)
						else:
							plt.savefig('A10_071/UVIT_%s.pdf' % obj)
					niters+=1
				print('\t...box generation complete.')
				#f.close()
			image.close()
'''
