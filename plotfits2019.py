import Analysis
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import os

run14 = Analysis.Run(np.load('2014_Raw.npy'), np.load('2014_Prune_strong.npy'), Analysis.bval2014, np.load('2014bf_new_prune_edit.npy'))
run11 = Analysis.Run(np.load('2011_Raw.npy'), np.load('2011_Prune_strong_flat.npy'), Analysis.bval2011, np.load('2011bf_new_prune_flat_edit.npy'))

i14 = run14.prune_data
i11 = run11.prune_data

print('Data loaded')

os.chdir('fitcheck2019')

print('Moved Dir')

print('2014 Data')
fig = plt.figure()
n = 0
for i in i14:
	k = 0
	b = Analysis.bval2014[n]
	for j in i.fits:
		i.plot()
		j.plot(False)
		plt.title('B:{},{}'.format(b,round((j.phase_r-j.phase_l)/np.pi%2,3)))
		fig.savefig('2014 {}_{}'.format(n,k))
		plt.clf()
		k += 1
	n += 1

print('2011 Data')
n = 0
for i in i11:
	k = 0
	b = Analysis.bval2011[n]
	for j in i.fits:
		i.plot()
		j.plot(False)
		plt.title('B:{},{}'.format(b,round((j.phase_r-j.phase_l)/np.pi%2,3)))
		fig.savefig('2011 {}_{}'.format(n,k))
		plt.clf()
		k += 1
	n += 1

print('Done')
