import numpy as np
import scipy.special as sps


def uniques(lst):
    uniquelist = list()
    for item in lst:
        if item in uniquelist:
            continue
        uniquelist.append(item)
    return uniquelist


def file_analyser(ffile):
    file = np.load(ffile)
    kk1s = file['k1s']
    kk1s = uniques(kk1s)
    kk2s = file['k2s']
    kk2s = uniques(kk2s)
    cchisqhere = file['chisq']
    cchisqhere = cchisqhere.reshape((len(kk1s), len(kk2s)))
    return kk1s, kk2s, cchisqhere.T


def plot_contours(ffig, kk1s, kk2s, cchisq):
    aax = ffig.add_subplot(111)
    k1grid, k2grid = np.meshgrid(kk1s, kk2s)
    contour = aax.contourf(k1grid, k2grid, cchisq, levels=50, cmap='inferno')
    ffig.colorbar(contour, ax=aax)
    return aax


def plot_uncertainty(ffig, kk1s, kk2s, cchisq, dof):
    chisqmin = np.min(cchisq)
    aax = ffig.add_subplot(111)
    p = 1-sps.gammainc(dof/2, cchisq/chisqmin/2)
    k1grid, k2grid = np.meshgrid(kk1s, kk2s)
    contour = aax.contourf(k1grid, k2grid, p, levels=50, cmap='inferno')
    ffig.colorbar(contour, ax=aax)
    return aax


def get_minimum(kk1s, kk2s, cchisq):
    idx = list(np.unravel_index(np.argmin(cchisq), cchisq.shape))
    mink1 = kk1s[idx[1]]
    mink2 = kk2s[idx[0]]
    return mink1, mink2


def mark_minimum(aax, kk1s, kk2s, cchisq):
    mink1, mink2 = get_minimum(kk1s, kk2s, cchisq)
    print(mink1, mink2)
    aax.plot([mink1], [mink2], 'ro', ms=5, label=r'$\chi^2_{red,min}$')
    return mink1, mink2


