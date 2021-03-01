
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

prim_purefile = glob.glob('9_Sgr/fd3/rangefinalprimary_norm.txt')[0]
sec_purefile = glob.glob('9_Sgr/fd3/rangefinalsecondary_norm.txt')[0]
prim_pure = np.loadtxt(prim_purefile).T
sec_pure = np.loadtxt(sec_purefile).T
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
plt.figure()
plt.plot(prim_pure[0], prim_pure[1]+0.1, label='fd3_prim_old')
plt.plot(prim[0], prim[1]+0.1, 'b', label='fd3_prim_new')


plt.grid()
sf.plot_line_list(y_text=1.3)
plt.plot(sec_pure[0], sec_pure[1], label='fd3_sec_old')
plt.plot(sec[0], sec[1], 'r', label='fd3_sec_new')
# axs[1].plot(sec[0], running_average(sec[1], av), label='{}-point running average'.format(av))
# axs[1].plot(model['wave'], model['norm_flux'], label='TLUSTY 15kK')
plt.legend()
plt.tight_layout()
plt.show()
