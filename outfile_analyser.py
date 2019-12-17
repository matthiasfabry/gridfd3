import glob
import sys

import matplotlib.pyplot as plt
import numpy as np


def uniques(lst):
    uniquelist = list()
    for item in lst:
        if item in uniquelist:
            continue
        uniquelist.append(item)
    return uniquelist


def file_analyser(ffiles):
    firstfile = np.load(ffiles[0])
    kk1s = firstfile['k1s']
    kk1s = uniques(kk1s)
    kk2s = firstfile['k2s']
    kk2s = uniques(kk2s)
    chisqhere = firstfile['chisq']
    for i in range(1, len(ffiles)):
        chisqhere += np.load(ffiles[i])['chisq']

    chisqhere = chisqhere.reshape((len(kk1s), len(kk2s)))/len(ffiles)
    return kk1s, kk2s, chisqhere.T


def plot_contours(kk1s, kk2s, cchisq):
    fig = plt.figure(figsize=(9, 6))
    aax = fig.add_subplot(111)
    k1grid, k2grid = np.meshgrid(kk1s, kk2s)
    contour = aax.contourf(k1grid, k2grid, cchisq, levels=50, cmap='inferno')

    fig.colorbar(contour, ax=aax)
    return aax


def mark_minimum(aax, kk1s, kk2s, cchisq):
    idx = list(np.unravel_index(np.argmin(cchisq), cchisq.shape))
    mink1 = kk1s[idx[1]]
    mink2 = kk2s[idx[0]]
    print(mink1, mink2)
    aax.plot([mink1], [mink2], 'ro', ms=10)


plt.rc('text', usetex=True)
plt.rc('font', size=16)

folder = None
try:
    folder = sys.argv[1]
except IndexError:
    print("Give a folder to work on!")
    exit()

lines = dict()
lines['Hzeta'] = (8.2630, 8.2685)
# lines['Hepsilon'] = (8.2845, 8.2888)
lines['HeI+II4026'] = (8.2990, 8.302)
lines['Hdelta'] = (8.3170, 8.3215)
# lines['SiIV4116'] = (8.3215, 8.3238)
lines['HeII4200'] = (8.3412, 8.3444)
lines['Hgamma'] = (8.3730, 8.3785)
lines['HeI4471'] = (8.4047, 8.4064)
lines['HeII4541'] = (8.4195, 8.4226)
lines['NV4604+4620'] = (8.4338, 8.4390)
lines['HeII4686'] = (8.4510, 8.4534)
lines['Hbeta'] = (8.4860, 8.4920)
lines['HeII5411'] = (8.5940, 8.5986)
lines['OIII5592'] = (8.6281, 8.6300)
# lines['CIII5696'] = (8.6466, 8.6482)
# lines['FeII5780'] = (8.6617, 8.6627)
lines['CIV5801'] = (8.6652, 8.6667)
lines['CIV5812'] = (8.6668, 8.6685)
lines['HeI5875'] = (8.6777, 8.6794)
files = list()
title = r"$\chi^2_\nu$"
for key in lines.keys():
    files.append(glob.glob(folder + '/**/chisq{}.npz'.format(key))[0])
    title += " ${}$".format(key)
k1s, k2s, chisq = file_analyser(files)
ax = plot_contours(k1s, k2s, chisq)
plottitle = title
if len(lines) > 5:
    plottitle = r"$\chi^2_\nu$ multiple lines"
ax.set_title(plottitle)
ax.set_xlabel(r'$K_1(km/s)$')
ax.set_ylabel(r'$K_2(km/s)$')
mark_minimum(ax, k1s, k2s, chisq)
plt.tight_layout()
plt.savefig(folder + '/{}.png'.format(title).replace("$", "").replace('\\', '').replace(' ', ''))
plt.show()
