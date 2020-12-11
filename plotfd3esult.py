
import sys
import glob
import matplotlib.pyplot as plt
import numpy as np
sys.path.extend(['/Users/matthiasf/data/spectra'])
# noinspection PyUnresolvedReferences
import spectralFunctions as sf
import astropy.io.fits as fits

#%%
folder = None
try:
    folder = sys.argv[1]
except IndexError:
    print('no object')
    exit()


primfile = glob.glob(folder + '/*primary_norm.txt')[0]
secfile = glob.glob(folder + '/*secondary_norm.txt')[0]
prim = np.loadtxt(primfile).T
sec = np.loadtxt(secfile).T

# prim_purefile = glob.glob(folder + '/rangeprimary.txt')[0]
# sec_purefile = glob.glob(folder + '/rangesecondary.txt')[0]
# prim_pure = np.loadtxt(prim_purefile).T
# sec_pure = np.loadtxt(sec_purefile).T
# model = fits.open('/Users/matthiasf/data/spectra/simulations/HD130298/secondary/BG15000g400v2.vis.fits')['NORM_SPECTRUM'].data


def running_average(fluxes, num):
    avgs = np.zeros(len(fluxes))
    lst = list()
    for i in range(len(fluxes)):
        lst.append(fluxes[i])
        avgs[i] = np.average(np.array(lst))
        if i > num:
            lst.pop(0)
    return avgs


av = 10
fig, axs = plt.subplots(2, sharex='all', sharey='all')
# axs[0].plot(prim_pure[0], prim_pure[1], label='fd3_prim_raw')
axs[0].plot(prim[0], prim[1], 'b', label='fd3_prim')

axs[0].legend()
axs[0].grid()
sf.plot_line_list(ax=axs[0], y_text=1.3)
# axs[1].plot(sec_pure[0], sec_pure[1], label='fd3_sec_raw')
axs[1].plot(sec[0], sec[1], 'r', label='fd3_sec')
axs[1].plot(sec[0], running_average(sec[1], av), label='{}-point running average'.format(av))
# axs[1].plot(model['wave'], model['norm_flux'], label='TLUSTY 15kK')
sf.plot_line_list(ax=axs[1], y_text=1.3)
axs[1].legend()
axs[1].grid()
fig.tight_layout()
plt.show()
