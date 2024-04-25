#!/Users/sargas/miniconda3/bin/python

'''
This code executes all the aperture photometry on the UVIT fields, including comparisons using GALEX apertures and magnitudes.

First the code does a comparison to all point source and extended magnitudes from GALEX that fall within each UVIT pointing. Point source magnitudes are measured within 5 UVIT FWHM and background-subtracted using the mean background measured in an annulus spanning 7-10 FWHM. Point source photometry is performed on the science image with background included, as the local PSF background will be dependent on both actual background and any light from nearby sources and this must be accounted for in the errors.

The magnitudes for extended sources are measured using those provided from Boselli et al. This means that if an object was not detected in GALEX, it will not be measured in the UVIT imaging. However, we can still check if there are objects in the GALEX imaging that were *not* detected with UVIT.

All measured magnitdes and errors, plus other values (mean background for each source, detection limit, area of the source aperture, etc.) are saved in GALEXPhotometry/*galex_match.dat. A correspond plot for each field, *_galex_sources.pdf, shows all detected GALEX sources in blue (extended) or orange (PSF) on the UVIT field (any non-detections are enclosed in gray). Lastly, the figure *galex_mags.pdf shows the difference between old GALEX FUV magnitdes and new UVIT photometry (measured with the above procedure) as a function of UVIT magnitude.

A very similar process is repeated to measure photometry for all NGVS objects. Here the apertures for extended sources are based on the ellipse shapes at 1 r_e from the optical NGVS data. The radii have been scaled by a factor of 3.5 as visual inspections suggested that this would enclose all UV signal in the UVIT imaging. UCDs and GCs are measured as point sources and have the same treatment as GALEX point sources above. Additionally, for galaxies, a central point source is measured to investigate the UV flux in the core (or nuclear star cluster, if present). As with other point sources, the background is measured in a local annulus and should (hopefully) account for contaminating light from the extended galaxy body.

All measured magnitudes and other properties are saved in NGVSPhotometry/*_ngvs_match.dat. Again, a *_ngvs_sources.pdf figure shows the apertures overlaid on the UVIT field, color-coded according to type of object and galaxy subclass.

A master file, ngvs_total_matches.dat, records the total and detected numbers of galaxies, UCDs, and GCs in each UVIT field. 
'''

import glob, os
import numpy as np
from astropy.convolution import Gaussian2DKernel
from astropy.io import fits
from astropy.stats import gaussian_fwhm_to_sigma, SigmaClip
from astropy.table import QTable, vstack
from astropy.convolution import convolve
import astropy.units as u
from astropy.visualization import make_lupton_rgb, SqrtStretch, ImageNormalize, simple_norm
import astropy.wcs as wcs
import astroimtools
from astropy.coordinates import angular_separation, Angle, SkyCoord
import astropy
import pandas
import random
import math
import csv

import photutils
from photutils.background import Background2D, MeanBackground, SExtractorBackground
from photutils.segmentation import detect_sources, deblend_sources, SourceCatalog, SegmentationImage#, source_properties (new API)
from photutils.utils import calc_total_error, circular_footprint
from photutils.aperture import SkyEllipticalAperture, SkyEllipticalAnnulus, SkyCircularAperture, SkyCircularAnnulus
from regions import Regions, PixCoord, EllipseSkyRegion, CircleSkyRegion
from photutils.aperture import aperture_photometry, ApertureStats

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as patches

# set up a kernal to blur the UVIT image for aesthetic display purposes
# 3 pixel blur
sigma = 3 * gaussian_fwhm_to_sigma  
kernel = Gaussian2DKernel(sigma)#, x_size=3, y_size=3)
kernel.normalize()

# create a mask to account for the padding to create a square image around the circular UVIT aperture
pixels = 14*60./0.417 # number of pixels in 14' given pixel scale
mask = astroimtools.circular_footprint(pixels)

pada = int((4800 - 2.0*pixels)/2.0)
padb = pada #- 1

foot = np.pad(mask, [(pada, padb), (pada, padb)], mode='constant', constant_values=0)
coverage_mask = (foot == 0)

# define a 1.5" FWHM gaussian to represent point sources in the UVIT data
uvitsigma = 1.5 * gaussian_fwhm_to_sigma 

# define a few other properties, useful for calculating point source mag limit later
zp = 17.771
radius = 1. / 0.417 # 1" radius in pixels
Npix = (radius**2 * np.pi)**0.5 # area of 1" aperture in pixels to get number of pixels inside

# get various properties of UVIT fields, mainly the central coordinates for each one
fields, exptime, obj, targ, prog = np.genfromtxt('uvit_header_info.dat', usecols=(0, 3, 4, 5, 6), dtype=str, unpack=True)
xc, yc = np.genfromtxt('uvit_header_info.dat', usecols=(1, 2), unpack=True)

# read in GALEX data and clean it so that we only select GALEX objects with detected FUV magnitudes and good photometry
hdu1 = fits.open('galex_extended_sources.fit')
galex_extended = hdu1[1].data
galex_extended = galex_extended[galex_extended['DEJ2000'] <= 90.]
galex_extended = galex_extended[galex_extended['FUVmag'] > 0.]

hdu2 = fits.open('galex_point_sources.fit')
galex_points = hdu2[1].data
galex_points = galex_points[galex_points['FUVmag'] > 0.]
galex_points = galex_points[galex_points['Fpf'] == 0.]
galex_points = galex_points[galex_points['Fsf'] == 0.]

# create arrays of SkyCoords for each catalog to streamline target matching with UVIT
ext_catalog = SkyCoord(ra=galex_extended['RAJ2000']*u.degree, dec=galex_extended['DEJ2000']*u.degree, frame='icrs')
psf_catalog = SkyCoord(ra=galex_points['RAJ2000']*u.degree, dec=galex_points['DEJ2000']*u.degree, frame='icrs')

for i,field in enumerate(fields):
	# open all necessary fits images for each UVIT pointing
	try:
			source = fits.open('science_imgs/%s_sci.fits' % field)
			bkgimg = fits.open('science_imgs/%s_bkg.fits' % field)
			bkgrms = fits.open('science_imgs/%s_rms.fits' % field)
			psfimg = fits.open('science_imgs/%s_bkg_psf.fits' % field)
			psfrms = fits.open('science_imgs/%s_rms_psf.fits' % field)
	except FileNotFoundError:
		continue

	imwcs = wcs.WCS(source[0].header, source)
	datasub = source[0].data
	psfsub = datasub + bkgimg[0].data - psfimg[0].data
	data = datasub + bkgimg[0].data
	convolved_data = convolve(datasub, kernel)
	error = calc_total_error(datasub, bkgrms[0].data, effective_gain=0)
	psferror = calc_total_error(psfsub, psfrms[0].data, effective_gain=0)
	masked = np.ma.masked_where(foot == 0, bkgrms[0].data)

	# 3 sigma point source mag limit for this UVIT field
	lim3sig = -2.5 * np.log10(3. * masked.mean() * Npix**0.5) + zp
	maxflux = 3. * masked.mean() * Npix**0.5

	scalarc = SkyCoord(ra=xc[i]*u.deg, dec=yc[i]*u.deg, frame='icrs') # center of UVIT field

	# find all extended GALEX sources within 13' of the field center
	d2d = scalarc.separation(ext_catalog)
	catalogmsk = d2d < 13.*u.arcmin
	idxcatalog = np.where(catalogmsk)[0]

	fig1 = plt.figure(1, figsize=(5.25, 3.25))
	ax1 = plt.subplot()

	fig_sky = plt.figure(2, figsize=(7.25, 7.25))
	ax_sky = plt.subplot(projection=imwcs)
	ax_sky.imshow(np.ma.array(convolved_data, mask=(foot == 0)), origin='lower', cmap='gist_gray_r', interpolation='nearest', norm=simple_norm(np.ma.array(convolved_data, mask=(foot == 0)), 'power', power=1.5, percent=99.5))

	f = open('GALEXphotometry/%s_galex_match.dat' % field, 'w')
	f.write('#RA\t\tDec\t\tSource\tUVIT_lim\tFUVmag_UVIT\terr\tFUVmag_GALEX\terr\tUVIT t_exp\tGALEX t_exp\tAvg bkg\t\tArea\n')

	for j in idxcatalog:
		# create regions to display the GALEX object on the UVIT image
		ellip = EllipseSkyRegion(ext_catalog[j], (2. * galex_extended['a'][j]) * u.arcsec, (2. * galex_extended['b'][j]) * u.arcsec, (180. - galex_extended['PA'][j]) * u.deg)
		ellip_tmp = ellip.to_pixel(imwcs)
		
		# create an aperture matching the GALEX region, plus an annulus just beyond it for background estimation
		aperture = SkyEllipticalAperture(ext_catalog[j], galex_extended['a'][j] * u.arcsec, galex_extended['b'][j] * u.arcsec, (90. - galex_extended['PA'][j]) * u.deg)
		annulus = SkyEllipticalAnnulus(ext_catalog[j], galex_extended['a'][j] * u.arcsec, (2.*galex_extended['a'][j]) * u.arcsec, (2.*galex_extended['b'][j]) * u.arcsec, galex_extended['b'][j] * u.arcsec, (90. - galex_extended['PA'][j]) * u.deg)

		aperture_tmp = aperture.to_pixel(imwcs)
		annulus_tmp = annulus.to_pixel(imwcs)

		# measure the average background in the annulus using the estimated background map
		aperstats = ApertureStats(bkgimg[0].data, aperture_tmp, mask=coverage_mask, error=bkgrms[0].data)
		bkg_mean = aperstats.mean

		phot_table = aperture_photometry(datasub, aperture_tmp, mask=coverage_mask, error=error)
		aperture_area = aperture_tmp.area_overlap(data, mask=coverage_mask)
		total_bkg = bkg_mean * aperture_area
		phot_bkgsub = phot_table['aperture_sum']# - total_bkg
		source_flux = phot_bkgsub[0]
		flux_err = phot_table['aperture_sum_err'][0]

		if source_flux > maxflux:
			maxflux = source_flux

		if source_flux <= 0.:
			patch = ellip_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='0.2', linestyle='--', lw=1.5)
			f.write('%f\t%f\tE\t%.2f\t\t-100.0\t\t-100.0\t%.3f\t\t%.3f\t%.1f\t\t%.1f\t\t%e\t%.2f\n' % (ext_catalog[j].ra.value, ext_catalog[j].dec.value, lim3sig, galex_extended['FUVmag'][j], galex_extended['e_FUVmag'][j], exptime[i], galex_extended['Fexp'][j], bkg_mean, aperture_area))
			ax1.errorbar(galex_extended['FUVmag'][j], 1.8, yerr=np.sqrt(galex_extended['e_FUVmag'][j]**2 + (1.086 * flux_err / source_flux)**2), xerr=galex_extended['e_FUVmag'][j], ecolor='0.8', elinewidth=1, marker='o', mfc='0.2', mec='none', linestyle='none', alpha=0.6)

		else:
			patch = ellip_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='tab:blue', linestyle='-', lw=1.5)
			f.write('%f\t%f\tE\t%.2f\t\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t%.1f\t\t%.1f\t\t%e\t%.2f\n' % (ext_catalog[j].ra.value, ext_catalog[j].dec.value, lim3sig, -2.5 * np.log10(source_flux) + zp, 1.086 * flux_err / source_flux, galex_extended['FUVmag'][j], galex_extended['e_FUVmag'][j], exptime[i], galex_extended['Fexp'][j], bkg_mean, aperture_area))
			ax1.errorbar(-2.5 * np.log10(source_flux) + zp, -2.5 * np.log10(source_flux) + zp - galex_extended['FUVmag'][j], yerr=np.sqrt(galex_extended['e_FUVmag'][j]**2 + (1.086 * flux_err / source_flux)**2), xerr=1.086 * flux_err / source_flux, ecolor='0.8', elinewidth=1, marker='o', mfc='tab:blue', mec='none', linestyle='none', alpha=0.6)

	d2d = scalarc.separation(psf_catalog)
	catalogmsk = d2d < 13.*u.arcmin
	idxcatalog = np.where(catalogmsk)[0]

	for j in idxcatalog:
		circ = CircleSkyRegion(psf_catalog[j], 5.0 * u.arcsec)
		circ_tmp = circ.to_pixel(imwcs)

		aperture = SkyCircularAperture(psf_catalog[j], (5.0 * uvitsigma) * u.arcsec)
		annulus = SkyCircularAnnulus(psf_catalog[j], (7.0 * uvitsigma) * u.arcsec, (10.0 * uvitsigma) * u.arcsec)

		aperture_tmp = aperture.to_pixel(imwcs)
		annulus_tmp = annulus.to_pixel(imwcs)

		aperstats = ApertureStats(data, annulus_tmp, mask=coverage_mask, error=psferror)
		bkg_mean = aperstats.mean

		phot_table = aperture_photometry(data, aperture_tmp, mask=coverage_mask, error=psferror)
		aperture_area = aperture_tmp.area_overlap(data, mask=coverage_mask)
		total_bkg = bkg_mean * aperture_area
		phot_bkgsub = phot_table['aperture_sum'] - total_bkg
		source_flux = phot_bkgsub[0]
		flux_err = phot_table['aperture_sum_err'][0]

		if source_flux > maxflux:
			maxflux = source_flux

		if source_flux <= 0.:
			patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='0.2', linestyle='--', lw=0.8)
			ax1.errorbar(galex_points['FUVmag'][j], 1.8, yerr=np.sqrt(galex_points['e_FUVmag'][j]**2 + (1.086 * flux_err / source_flux)**2), xerr=galex_points['e_FUVmag'][j], ecolor='0.8', elinewidth=1, marker='.', mfc='0.2', mec='none', linestyle='none', alpha=0.6)
			f.write('%f\t%f\tP\t%.2f\t\t-100.0\t\t-100.0\t%.3f\t\t%.3f\t%.1f\t\t%.1f\t\t%e\t%.2f\n' % (psf_catalog[j].ra.value, psf_catalog[j].dec.value, lim3sig, galex_points['FUVmag'][j], galex_points['e_FUVmag'][j], exptime[i], galex_points['Fexp'][j], bkg_mean, aperture_area))

		else:
			patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='tab:orange', linestyle='-', lw=0.8)
			ax1.errorbar(-2.5 * np.log10(source_flux) + zp, -2.5 * np.log10(source_flux) + zp - galex_points['FUVmag'][j], yerr=np.sqrt(galex_points['e_FUVmag'][j]**2 + (1.086 * flux_err / source_flux)**2), xerr=1.086 * flux_err / source_flux, ecolor='0.8', elinewidth=1, marker='.', mfc='tab:orange', mec='none', linestyle='none', alpha=0.6)
			f.write('%f\t%f\tP\t%.2f\t\t%.3f\t\t%.3f\t%.3f\t\t%.3f\t%.1f\t\t%.1f\t\t%e\t%.2f\n' % (psf_catalog[j].ra.value, psf_catalog[j].dec.value, lim3sig, -2.5 * np.log10(source_flux) + zp, 1.086 * flux_err / source_flux, galex_points['FUVmag'][j], galex_points['e_FUVmag'][j], exptime[i], galex_points['Fexp'][j], bkg_mean, aperture_area))
	
	ax1.set_xlabel('UVIT FUV mag')
	ax1.set_ylabel('(UVIT - GALEX) FUV mag')
	ax1.set_xlim((lim3sig+0.5, -2.5 * np.log10(maxflux) + zp - 0.5))
	ax1.set_ylim((-3, 3))
	ax1.plot(np.linspace(-10, 10)*0 + lim3sig, np.linspace(-10, 10), linestyle='--', color='0.2', alpha=0.8)
	ax1.plot(np.linspace(0, 30), np.linspace(0, 30)*0., linestyle='--', color='0.2', alpha=0.8, zorder=0)
	fig1.tight_layout()
	fig_sky.tight_layout()
	ax_sky.set_xlabel('Right Ascension (J2000)')
	ax_sky.set_ylabel('Declination (J2000)')
	fig1.savefig('GALEXphotometry/%s_galex_mags.pdf' % field)
	fig_sky.savefig('GALEXphotometry/%s_galex_sources.pdf' % field)
	source.close()
	f.close()
	plt.close(fig1)
	plt.close(fig_sky)
	bkgimg.close()
	bkgrms.close()

# create SkyCoords and array of object types for the various categories of NGVS objects
xs = []
ys = []
names = []
colors = []
aas = []
bbs = []
thetas = []
flags = []
with open('/Users/sargas/Downloads/max/ngvs_gals_class0.reg') as file:
	for line in file:
		if 'ellipse' not in line:
			pass
		else:
			tmpcolor = line.split('color = ')[-1][:-1]
			tmp = line[line.find('(')+1:line.find(')')]
			params = tmp.split(',')
			#center_sky = SkyCoord(float(params[0]), float(params[1]), unit='deg', frame='icrs')
			xs.append(float(params[0]))
			ys.append(float(params[1]))
			names.append(line[line.find('{')+1:line.find('}')])
			colors.append(tmpcolor.split(' ')[0])
			aas.append(float(params[2].split('"')[0]))
			bbs.append(float(params[3].split('"')[0]))
			thetas.append(180. - float(params[4]))
			flags.append('gal0')
			#region_sky = EllipseSkyRegion(center=center_sky, height=(7. * float(params[2].split('"')[0])) * u.arcsec, width=(7. * float(params[3].split('"')[0])) * u.arcsec, angle=(90. + float(params[4])) * u.deg, visual={'fontsize':10, 'color':tmpcolor.split(' ')[0]})
			#regions.append(region_sky)

with open('/Users/sargas/Downloads/max/ngvs_gals_class1.reg') as file:
	for line in file:
		if 'ellipse' not in line:
			pass
		else:
			tmpcolor = line.split('color=')[-1][:-1]
			tmp = line[line.find('(')+1:line.find(')')]
			params = tmp.split(',')
			#center_sky = SkyCoord(float(params[0]), float(params[1]), unit='deg', frame='icrs')
			xs.append(float(params[0]))
			ys.append(float(params[1]))
			names.append(line[line.find('{')+1:line.find('}')])
			colors.append(tmpcolor.split(' ')[0])
			aas.append(float(params[2].split('"')[0]))
			bbs.append(float(params[3].split('"')[0]))
			thetas.append(180. - float(params[4]))
			flags.append('gal1')
			#region_sky = EllipseSkyRegion(center=center_sky, height=(7. * float(params[2].split('"')[0])) * u.arcsec, width=(7. * float(params[3].split('"')[0])) * u.arcsec, angle=(90. + float(params[4])) * u.deg, visual={'fontsize':10, 'color':tmpcolor.split(' ')[0]})
			#regions_gal1.append(region_sky)

with open('/Users/sargas/Downloads/max/ngvs_gals_class2.reg') as file:
	for line in file:
		if 'ellipse' not in line:
			pass
		else:
			tmpcolor = line.split('color=')[-1][:-1]
			tmp = line[line.find('(')+1:line.find(')')]
			params = tmp.split(',')
			#center_sky = SkyCoord(float(params[0]), float(params[1]), unit='deg', frame='icrs')
			xs.append(float(params[0]))
			ys.append(float(params[1]))
			names.append(line[line.find('{')+1:line.find('}')])
			colors.append(tmpcolor.split(' ')[0])
			aas.append(float(params[2].split('"')[0]))
			bbs.append(float(params[3].split('"')[0]))
			thetas.append(180. - float(params[4]))
			flags.append('gal2')
			#region_sky = EllipseSkyRegion(center=center_sky, height=(7. * float(params[2].split('"')[0])) * u.arcsec, width=(7. * float(params[3].split('"')[0])) * u.arcsec, angle=(90. + float(params[4])) * u.deg, visual={'fontsize':10, 'color':tmpcolor.split(' ')[0]})
			#regions_gal2.append(region_sky)

# open file and create reader
with open('/Users/sargas/Downloads/GCCAT/all_gcs.cat', 'r') as file:
	reader = csv.reader(file, delimiter=',', quotechar='"', skipinitialspace=True)
	# read header
	header = reader.__next__()
	# read rows, append values to lists
	for i,row in enumerate(reader):
		flags.append('gc')
		xs.append(float(row[0]))
		ys.append(float(row[1]))

ucdhdu = fits.open('/Users/sargas/Dropbox/Current/Data/liu2020_ucd_candidates.fit')
for i in range(len(ucdhdu[1].data['RAJ2000'])):
	xs.append(ucdhdu[1].data['RAJ2000'][i])
	ys.append(ucdhdu[1].data['DEJ2000'][i])
	flags.append('ucd')
	aas.append(ucdhdu[1].data['rh'][i] / 16500000. * 206265.) #convert pc to arcsec
	bbs.append(ucdhdu[1].data['rh'][i] / 16500000. * 206265.)
	thetas.append(0.0)
ucdhdu.close()

colors = [w.replace('#8EF', 'darkturquoise') for w in colors]
colors = [w.replace('#EF2', 'gold') for w in colors]
colors = [w.replace('#0F3', 'limegreen') for w in colors]

'''
with open('/Users/sargas/Downloads/max/ngvs_ucds.reg') as file:
	for line in file:
		if 'point' not in line:
			pass
		else:
			tmp = line[line.find('(')+1:line.find(')')]
			params = tmp.split(',')
			xs.append(float(params[0]))
			ys.append(float(params[1]))
			flags.append('ucd')
			#center_sky = SkyCoord(float(params[0]), float(params[1]), unit='deg', frame='icrs')
			#region_sky = CircleSkyRegion(center=center_sky, radius=10.0 * u.arcsec, visual={'color'--'magenta'})
			#regions_ucd.append(region_sky)
'''

ngvs_catalog = SkyCoord(ra=xs * u.degree, dec=ys * u.degree, frame='fk5')

ff = open('ngvs_total_matches.dat', 'w')
ff.write('#UVIT field\tMain obj.\tTot_gal\tN_det\tTot_UCD\tN_det\tTot_gc\tN_det\n')

for i, field in enumerate(field):
	ngals = 0
	nucds = 0
	ngcs = 0
	ngals_sub = 0
	nucds_sub = 0
	ngcs_sub = 0
	# open all necessary fits images for each UVIT pointing
	try:
			source = fits.open('science_imgs/%s_sci.fits' % field)
			bkgimg = fits.open('science_imgs/%s_bkg.fits' % field)
			bkgrms = fits.open('science_imgs/%s_rms.fits' % field)
			segmap = fits.open('science_imgs/%s_segmap.fits' % field)
			psfimg = fits.open('science_imgs/%s_bkg_psf.fits' % field)
			psfrms = fits.open('science_imgs/%s_rms_psf.fits' % field)
			segmap_psf = fits.open('science_imgs/%s_segmap_psf.fits' % field)
	except FileNotFoundError:
		continue

	imwcs = wcs.WCS(source[0].header, source)
	datasub = source[0].data
	psfsub = datasub + bkgimg[0].data - psfimg[0].data
	data = datasub + bkgimg[0].data
	convolved_data = convolve(datasub, kernel)
	error = calc_total_error(datasub, bkgrms[0].data, effective_gain=0)
	psferror = calc_total_error(psfsub, psfrms[0].data, effective_gain=0)
	masked = np.ma.masked_where(foot == 0, bkgrms[0].data)

	segm = SegmentationImage(segmap[0].data)
	segmpsf = SegmentationImage(segmap_psf[0].data)

	cat = SourceCatalog(datasub, segm, error=error, background=bkgimg[0].data, wcs=imwcs)
	tbl = cat.to_table()
	source_coords = []
	for k in range(len(tbl['xcentroid'].data)):
		source_coords.append([tbl['xcentroid'].data[k], tbl['ycentroid'].data[k]])

	cat = SourceCatalog(psfsub, segmpsf, error=psferror, background=psfimg[0].data, wcs=imwcs)
	tbl = cat.to_table()
	for k in range(len(tbl['xcentroid'].data)):
		source_coords.append([tbl['xcentroid'].data[k], tbl['ycentroid'].data[k]])
	source_coords = np.array(source_coords)

	# 3 sigma point source mag limit for this UVIT field
	lim3sig = -2.5 * np.log10(3. * masked.mean() * Npix**0.5) + zp
	minflux = 3. * masked.mean() * Npix**0.5
	maxflux = 3. * masked.mean() * Npix**0.5

	scalarc = SkyCoord(ra=xc[i]*u.deg, dec=yc[i]*u.deg, frame='icrs')

	fig1 = plt.figure(1, figsize=(5.25, 3.25))
	ax1 = plt.subplot()

	fig_sky = plt.figure(2, figsize=(7.25, 7.25))
	ax_sky = plt.subplot(projection=imwcs)
	ax_sky.imshow(np.ma.array(convolved_data, mask=(foot == 0)), origin='lower', cmap='gist_gray_r', interpolation='nearest', norm=simple_norm(np.ma.array(convolved_data, mask=(foot == 0)), 'power', power=1.5, percent=99.5))

	d2d = scalarc.separation(ngvs_catalog)
	catalogmsk = d2d < 13.*u.arcmin
	idxcatalog = np.where(catalogmsk)[0]

	f = open('NGVSphotometry/%s_ngvs_match.dat' % field, 'w')
	f.write('#RA\t\tDec\t\tSource\t3sig_maglim\tmag_FUV\terr\tinner_mag\terr\tUVIT t_exp\tAvg bkg(per pix)\tArea(pix)\tArea(")\tmu_FUV\t3sig_mulim\n')

	for j in idxcatalog:
		if 'gal' in flags[j]:
			# we have an extended source, create an elliptical aperture
			ellip = EllipseSkyRegion(ngvs_catalog[j], (7. * aas[j]) * u.arcsec, (7. * bbs[j]) * u.arcsec, (180. - thetas[j]) * u.deg)
			ellip_tmp = ellip.to_pixel(imwcs)
			newx, newy = ngvs_catalog[j].to_pixel(imwcs)

			# create an aperture matching the NGVS region, plus an annulus just beyond it for background estimation
			aperture = SkyEllipticalAperture(ngvs_catalog[j], (3.5 * aas[j]) * u.arcsec, (3.5 * bbs[j]) * u.arcsec, (90. - thetas[j]) * u.deg)
			annulus = SkyEllipticalAnnulus(ngvs_catalog[j], (3.5 * aas[j]) * u.arcsec, (7.* aas[j]) * u.arcsec, (7. * bbs[j]) * u.arcsec, (3.5 * bbs[j]) * u.arcsec, (90. - thetas[j]) * u.deg)

			aperture_tmp = aperture.to_pixel(imwcs)
			annulus_tmp = annulus.to_pixel(imwcs)

			xmin, xmax, ymin, ymax = [num for num in aperture_tmp.bbox.extent]
			#segmap[0].data[ymin:ymax, xmin:xmax]
			ll = np.array([xmin, ymin])
			ur = np.array([xmax, ymax])
			inidx = np.all(np.logical_and(ll <= source_coords, source_coords <= ur), axis=1)

			# measure the average background in the annulus using the estimated background map
			aperstats = ApertureStats(bkgimg[0].data, aperture_tmp, mask=coverage_mask, error=bkgrms[0].data)
			bkg_mean = aperstats.mean

			phot_table = aperture_photometry(datasub, aperture_tmp, mask=coverage_mask, error=error)
			aperture_area = aperture_tmp.area_overlap(data, mask=coverage_mask) # in pixels
			area_arc = aperture_area * 0.417**2
			mu3sig = -2.5 * np.log10(3. * masked.mean() * aperture_area**0.5) + zp + 2.5*np.log10(area_arc)
			total_bkg = bkg_mean * aperture_area
			phot_bkgsub = phot_table['aperture_sum']# - total_bkg
			source_flux = phot_bkgsub[0]
			flux_err = phot_table['aperture_sum_err'][0]

			# also measure a central point source
			inner_aperture = SkyCircularAperture(ngvs_catalog[j], (5.0 * uvitsigma) * u.arcsec)
			inner_annulus = SkyCircularAnnulus(ngvs_catalog[j], (7.0 * uvitsigma) * u.arcsec, (10.0 * uvitsigma) * u.arcsec)
			inner_tmp = inner_aperture.to_pixel(imwcs)
			innann_tmp = inner_annulus.to_pixel(imwcs)
			xmin, xmax, ymin, ymax = [num for num in inner_tmp.bbox.extent]
			ll = np.array([xmin, ymin])
			ur = np.array([xmax, ymax])
			inidx = np.all(np.logical_and(ll <= source_coords, source_coords <= ur), axis=1)

			aperstats = ApertureStats(data, innann_tmp, mask=coverage_mask, error=bkgrms[0].data)
			inner_bkg_mean = aperstats.mean
			inner_table = aperture_photometry(data, inner_tmp, mask=coverage_mask, error=error)
			inner_area = inner_tmp.area_overlap(data, mask=coverage_mask)
			inner_arc = inner_area * 0.417**2
			#mu3sig = lim3sig + 2.5*np.log10(np.pi)
			inner_bkg = inner_bkg_mean * inner_area
			inner_bkgsub = inner_table['aperture_sum'] - inner_bkg
			inner_flux = inner_bkgsub[0]
			inner_err = inner_table['aperture_sum_err'][0]

			if source_flux > maxflux:
				maxflux = source_flux

			if ((source_flux < minflux) or (len(inidx) < 1)):# or -2.5 * np.log10(source_flux) + zp + 2.5*np.log10(area_arc) > mu3sig):
				ngals+=1
				patch = ellip_tmp.plot(ax=ax_sky, facecolor='none', edgecolor=colors[j], linestyle='--', lw=1.5)
				ax_sky.text(newx, newy+20., names[j], color=colors[j], horizontalalignment='center', verticalalignment='bottom', fontsize=10, fontname='Helvetica', usetex=False)
				if inner_flux > minflux:
					print('no galaxy but it has a center')
					f.write('%f\t%f\t%s\t%.2f\t\t-100.0\t-100.0\t%.3f\t\t%.3f\t%.1f\t\t%e\t%12.2f\t%12.2f\t-100.00\t%.2f\n' % (ngvs_catalog[j].ra.value, ngvs_catalog[j].dec.value, flags[j], lim3sig, -2.5 * np.log10(inner_flux) + zp, 1.086 * inner_err / inner_flux, exptime[i], bkg_mean, aperture_area, area_arc, mu3sig))


			else:
				ngals_sub+=1
				ngals+=1
				patch = ellip_tmp.plot(ax=ax_sky, facecolor='none', edgecolor=colors[j], linestyle='-', lw=1.5)
				ax_sky.text(newx, newy+20., names[j], color=colors[j], horizontalalignment='center', verticalalignment='bottom', fontsize=10, fontname='Helvetica', usetex=False)

				f.write('%f\t%f\t%s\t%.2f\t\t%.3f\t%.3f\t%.3f\t\t%.3f\t%.1f\t\t%e\t%12.2f\t%12.2f\t%.2f\t%.2f\n' % (ngvs_catalog[j].ra.value, ngvs_catalog[j].dec.value, flags[j], lim3sig, -2.5 * np.log10(source_flux) + zp, 1.086 * flux_err / source_flux, -2.5 * np.log10(inner_flux) + zp, 1.086 * inner_err / inner_flux, exptime[i], bkg_mean, aperture_area, area_arc, -2.5 * np.log10(source_flux) + zp + 2.5*np.log10(area_arc), mu3sig))

		else:
			# either a UCD or GC, so just do a circular aperture
			circ = CircleSkyRegion(ngvs_catalog[j], 5.0 * u.arcsec)
			circ_tmp = circ.to_pixel(imwcs)
			aperture = SkyCircularAperture(ngvs_catalog[j], (5.0 * uvitsigma) * u.arcsec)
			source_aperture = SkyCircularAperture(ngvs_catalog[j], uvitsigma * u.arcsec)
			annulus = SkyCircularAnnulus(ngvs_catalog[j], (7.0 * uvitsigma) * u.arcsec, (10.0 * uvitsigma) * u.arcsec)

			aperture_tmp = aperture.to_pixel(imwcs)
			source_tmp = source_aperture.to_pixel(imwcs)
			annulus_tmp = annulus.to_pixel(imwcs)
			xmin, xmax, ymin, ymax = [num for num in source_tmp.bbox.extent]
			ll = np.array([xmin, ymin])
			ur = np.array([xmax, ymax])
			inidx = np.all(np.logical_and(ll <= source_coords, source_coords <= ur), axis=1)

			aperstats = ApertureStats(data, annulus_tmp, mask=coverage_mask, error=psferror)
			bkg_mean = aperstats.mean

			phot_table = aperture_photometry(data, aperture_tmp, mask=coverage_mask, error=psferror)
			aperture_area = aperture_tmp.area_overlap(data, mask=coverage_mask)
			area_arc = aperture_area * 0.417**2
			#mu3sig = lim3sig + 2.5*np.log10(np.pi)
			total_bkg = bkg_mean * aperture_area
			phot_bkgsub = phot_table['aperture_sum'] - total_bkg
			source_flux = phot_bkgsub[0]
			flux_err = phot_table['aperture_sum_err'][0]

			if source_flux > maxflux:
				maxflux = source_flux

			if ((source_flux < minflux) or (len(inidx) < 1)):
				if flags[j] == 'gc':
					ngcs+=1
					#patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='yellow', linestyle='--', lw=1.)
				else:
					nucds+=1
					#patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='magenta', linestyle='--', lw=1.)

			else:
				if flags[j] == 'gc':
					ngcs_sub+=1
					ngcs+=1
					patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='gold', linestyle='-', lw=0.8)
					f.write('%f\t%f\tGC\t%.2f\t\t%.3f\t%.3f\t-100.0\t\t-100.0\t%.1f\t\t%e\t%12.2f\t%12.2f\t-100.0\t-100.0\n' % (ngvs_catalog[j].ra.value, ngvs_catalog[j].dec.value, lim3sig, -2.5 * np.log10(source_flux) + zp, 1.086 * flux_err / source_flux, exptime[i], bkg_mean, aperture_area, area_arc))

				else:
					nucds_sub+=1
					nucds+=1
					patch = circ_tmp.plot(ax=ax_sky, facecolor='none', edgecolor='magenta', linestyle='-', lw=0.8)
					f.write('%f\t%f\tUCD\t%.2f\t\t%.3f\t%.3f\t-100.0\t\t-100.0\t%.1f\t\t%e\t%12.2f\t%12.2f\t-100.0\t-100.0\n' % (ngvs_catalog[j].ra.value, ngvs_catalog[j].dec.value, lim3sig, -2.5 * np.log10(source_flux) + zp, 1.086 * flux_err / source_flux, exptime[i], bkg_mean, aperture_area, area_arc))
	fig_sky.tight_layout()
	ax_sky.set_xlabel('Right Ascension (J2000)')
	ax_sky.set_ylabel('Declination (J2000)')
	fig_sky.savefig('NGVSphotometry/%s_ngvs_sources.pdf' % field)
	source.close()
	f.close()
	plt.close(fig_sky)
	bkgimg.close()
	bkgrms.close()
	ff.write('%s\t\t%s\t\t%d\t%d\t%d\t%d\t%d\t%d\n' % (field[i], obj[i], ngals, ngals_sub, nucds, nucds_sub, ngcs, ngcs_sub))
ff.close()

