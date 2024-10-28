import numpy as np

import matplotlib.pyplot as plt

from pathlib import Path

root_folder = "~/dev/astrosat/UVIT_Analysis/data/A10_071/"

threshold_folder = root_folder + "thresholds/"

#exps = [300, 4000, 7500, 11000, 20000]
exps = [300]

# Read in all the backgrounds from random boxes and make histograms, check skew
for exp in exps:
	best = 1000.
	for backthresh in np.linspace(0.5, 3, num=51):
		boxnum, stuff = np.genfromtxt(threshold_folder +'threshold_%d_%.2f.dat' % (exp, backthresh), usecols=(0,3), unpack=True)
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


	boxnum, stuff = np.genfromtxt(threshold_folder + 'threshold_%d_1.00.dat' % exp, usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='0.5', linestyle=':', label=r'$1\,\sigma$')
	boxnum, stuff = np.genfromtxt(threshold_folder + 'threshold_%d_2.50.dat' % exp, usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='0.5', linestyle='--', label=r'$2.5\,\sigma$')

	boxnum, stuff = np.genfromtxt(threshold_folder + 'threshold_%d_%.2f.dat' % (exp, bestthresh), usecols=(0,3), unpack=True)
	plt.hist(stuff, bins=40, range=(0.5e-5, 4.75e-5), histtype='step', color='tab:blue', lw=1.5, zorder=10, label=r'$%.2f\,\sigma$' % bestthresh)
	globalbkg = np.mean(stuff)
	globalerr = np.std(stuff)
	print(globalbkg, globalerr, len(stuff))
	plt.legend()
	plt.xlabel(r'Background (counts sec$^{-1}$)')
	plt.ylabel(r'N')
	plt.tight_layout()
	plt.savefig(root_folder + 'threshold_%d.pdf' % exp)
	plt.clf()