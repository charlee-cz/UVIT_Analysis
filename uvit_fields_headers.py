#!/Users/sargas/miniconda3/bin/python

import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.coordinates import Longitude, Latitude
from astropy.wcs import WCS
import astropy.units as u
import glob, os
from astroquery.skyview import SkyView
from astropy.visualization.wcsaxes import SphericalCircle

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

#Define a convenience function that gets SkyView image for a particular survey, centre, and area on sky
def get_skyview(survey_name, ra, dec, size):
    hdu = SkyView.get_images("{0},{1}".format(ra, dec),
                             radius = 13*u.deg,
                             survey=survey_name, deedger="skyview.process.Deedger", scaling='linear')[0]
    image_header = hdu[0].header
    return (hdu)

'''
actual science time is RDCDTIME
center WCS position is CCVALD1 and CCVALD2? except not always because of course not
so if those values are 0, just convert center pixels (2400, 2400) to coords (or 2048 x 2048 for M87)
'''

exptimes = []
ids = []
targnames = []
ras = []
decs = []
prog = []

for fullpath in glob.iglob('/Users/sargas/Documents/UVIT/A*/*.fits', recursive=True):
	filename = os.path.basename(fullpath)
	# only work on the image files
	if 'EXPARRAY' not in filename:
		# ingore sextractor images
		if 'BaF2' in filename:
			with fits.open(fullpath) as hdul:
				exptimes = np.append(exptimes, hdul[0].header['RDCDTIME'])
				ids = np.append(ids, hdul[0].header['OBJECT'])
				targnames = np.append(targnames, hdul[0].header['TARGETID'])
				tmpra = hdul[0].header['CCVALD1']
				tmpdec = hdul[0].header['CCVALD2']
				if tmpra == 0:
					w = WCS(hdul[0].header)
					sky = w.pixel_to_world(2400, 2400)
					tmpra = sky.ra.degree
					tmpdec = sky.dec.degree
				ras = np.append(ras, tmpra)
				decs = np.append(decs, tmpdec)
			if 'A08' in filename:
				prog = np.append(prog, 'A08_003')
			else:
				prog = np.append(prog, 'A10_071')

with fits.open('/Users/sargas/Documents/UVIT/M87/M87_FUV_F2___MASTER.fits') as hdul:
	exptimes = np.append(exptimes, hdul[0].header['TOTITIME'])
	ids = np.append(ids, hdul[0].header['OBJECT'])
	targnames = np.append(targnames, hdul[0].header['TARGETID'])
	w = WCS(hdul[0].header)
	sky = w.pixel_to_world(2048, 2048)
	tmpra = sky.ra.degree
	tmpdec = sky.dec.degree
	ras = np.append(ras, tmpra)
	decs = np.append(decs, tmpdec)
	prog = np.append(prog, 'M87')

# sort UVIT fields by RA and Dec for new naming convention
temp = sorted(zip(ras, decs, ids, targnames, exptimes, prog), key=lambda x: (x[0], -x[1]))
newras, newdecs, newids, newtargs, newexps, newprog = map(list, zip(*temp))

# plot each field on sky
fig = plt.figure(figsize=(3.35, 3.65))
im = get_skyview('DSS', 187.7, 11.5, 30000)
imwcs = WCS(im[0].header, im)
ax = plt.subplot(projection=imwcs)
plt.imshow(im[0].data, origin='lower', cmap='gist_gray_r')

# overlay NGVS footprint
xs, ys = np.genfromtxt('/Users/sargas/Dropbox/Current/Data/footprint_coords.dat', unpack=True)
poly = Polygon([[xs[i], ys[i]] for i in range(len(xs))])
plt.plot(xs*u.deg, ys*u.deg, color='0.2', linestyle='-', linewidth=0.8, transform=ax.get_transform('icrs'), zorder=0)

# overlay GALEX pointings from Boselli+ 2011
xxs, yys, fuv_t = np.genfromtxt('/Users/sargas/Documents/UVIT/galex_virgo_fields.dat', usecols=(1, 2, 4), dtype=str, unpack=True)
fuv_t = fuv_t.astype(float)
has_fuv = np.where(fuv_t > 0.0)[0]

for i in range(len(xxs[has_fuv])):
	xtmp = Longitude(xxs[has_fuv][i] + ' hours')
	ytmp = Latitude(yys[has_fuv][i] + ' degrees')
	c = SphericalCircle((xtmp.degree*u.deg, ytmp.degree*u.deg), 1.2*u.deg, facecolor='tab:orange', edgecolor='none', linewidth=0., transform=ax.get_transform('icrs'), zorder=10, alpha=np.log10(fuv_t[has_fuv][i])/15.)
	ax.add_patch(c)


# plot UVIT pointings and also check which ones fall within NGVS area
infoot = []
for i in range(len(newras)):
	c = SphericalCircle((newras[i]*u.deg, newdecs[i]*u.deg), 14*u.arcmin, edgecolor='tab:blue', facecolor='none', linewidth=1.2, transform=ax.get_transform('icrs'), zorder=11)
	ax.add_patch(c)
	point = Point(newras[i], newdecs[i])
	infoot = np.append(infoot, poly.contains(point))

ax.set_xlim((65, 225))
ax.set_ylim((55, 235))
plt.xlabel(r'Right Ascension')
plt.ylabel(r'Declination')
plt.subplots_adjust(right=0.99, top=0.99, bottom=0.11, left=0.15)
plt.savefig('footprints.pdf')

f = open('uvit_header_info.dat', 'w')
f.write('#Field\t\tRA\t\tDec\t\tExpTime\t\tVCCtarget\tTargetID\tProgram\tInNGVSarea?\n')
for i in range(len(newras)):
	if infoot[i]:
		f.write('UVIT-%d\t\t%f\t%f\t%.2f\t%s\t%s\t%s\tTrue\n' % (i+1, newras[i], newdecs[i], newexps[i], newids[i], newtargs[i], newprog[i]))
	else:
		f.write('UVIT-%d\t\t%f\t%f\t%.2f\t%s\t%s\t%s\tFalse\n' % (i+1, newras[i], newdecs[i], newexps[i], newids[i], newtargs[i], newprog[i]))
f.close()



