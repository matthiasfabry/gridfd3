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
    ffile = np.load(ffile)
    kk1s = ffile['k1s']
    kk1s = uniques(kk1s)
    kk2s = ffile['k2s']
    kk2s = uniques(kk2s)
    cchisqhere = ffile['chisq']
    cchisqhere = cchisqhere.reshape((len(kk1s), len(kk2s)))
    return kk1s, kk2s, cchisqhere


def plot_contours(ffig, kk1s, kk2s, cchisq, ddof):
    aax = ffig.add_subplot(111)
    k2grid, k1grid = np.meshgrid(kk2s, kk1s)
    contour = aax.contourf(k1grid, k2grid, cchisq / ddof, levels=15, cmap='inferno')
    ffig.colorbar(contour, ax=aax)
    return aax


def plot_oneDee(ffig, kk2s, cchisq, ddof):
    axx = ffig.add_subplot(111)
    axx.plot(kk2s, cchisq / ddof)
    return axx


def plot_uncertainty(ffig, kk1s, kk2s, cchisq, ddof):
    factor = np.min(cchisq) / ddof
    aax = ffig.add_subplot(111)
    p = 1 - sps.gammainc(ddof / 2, (cchisq / factor) / 2)
    k2grid, k1grid = np.meshgrid(kk2s, kk1s)
    contour = aax.contourf(k1grid, k2grid, p, cmap='inferno')
    ffig.colorbar(contour, ax=aax)
    return aax


def plot_uncertainty_unscaled(ffig, kk1s, kk2s, cchisq, ddof):
    aax = ffig.add_subplot(111)
    p = 1 - sps.gammainc(ddof / 2, cchisq / 2)
    k2grid, k1grid = np.meshgrid(kk2s, kk1s)
    contour = aax.contourf(k1grid, k2grid, p, cmap='inferno')
    ffig.colorbar(contour, ax=aax)
    return aax


def get_minimum(kk1s, kk2s, cchisq):
    iidx = list(np.unravel_index(np.argmin(cchisq), cchisq.shape))
    mink1 = kk1s[iidx[0]]
    mink2 = kk2s[iidx[1]]
    return mink1, mink2


def get_min_idx(cchisq) -> np.ndarray:
    return np.array(np.unravel_index(np.argmin(cchisq), cchisq.shape))


def mark_minimum(aax, kk1s, kk2s, cchisq, label):
    mink1, mink2 = get_minimum(kk1s, kk2s, cchisq)
    aax.plot([mink1], [mink2], 'ro', ms=5, label=label)
    return mink1, mink2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import glob
    # noinspection PyUnresolvedReferences
    import plotsetup
    import sys

    folder = sys.argv[1]
    dof = 0
    lines = dict()
    # lines['Heta'] = (8.249, 8.2545)
    # lines['Hzeta'] = (8.263, 8.2685)
    # lines['FeI3923'] = (8.273, 8.2765)
    # lines['HeI+II4026'] = (8.2985, 8.3025)
    # lines['Hdelta'] = (8.3155, 8.3215)
    # lines['SiIV4116'] = (8.3215, 8.3238)
    # lines['HeI4143'] = (8.3280, 8.3305)
    # lines['HeII4200'] = (8.3410, 8.3445)
    # lines['Hgamma'] = (8.3720, 8.3785)
    # lines['NIII4379'] = (8.3835, 8.3857)
    # lines['HeI4387'] = (8.3855, 8.3875)
    # lines['HeI4471'] = (8.4045, 8.4065)
    # lines['HeII4541'] = (8.4190, 8.4230)
    lines['FeII4584'] = (8.4293, 8.4315)
    # lines['HeII4686'] = (8.4510, 8.4535)
    # lines['HeI4713'] = (8.4575, 8.4590)
    # lines['Hbeta'] = (8.4850, 8.4925)
    # lines['HeI5016'] = (8.5195, 8.5215)
    lines['FeII5167'] = (8.5495, 8.5513)
    lines['FeII5198'] = (8.5550, 8.5567)
    lines['FeII5233'] = (8.5621, 8.5639)
    lines['FeII5276'] = (8.5700, 8.5716)
    lines['FeII5316'] = (8.5778, 8.5793)
    # lines['HeII5411'] = (8.5935, 8.5990)
    # lines['FeII5780'] = (8.6617, 8.6629)
    # lines['CIV5801'] = (8.6652, 8.6667)
    # lines['CIV5812'] = (8.6668, 8.6685)
    # lines['HeI5875'] = (8.6775, 8.6795)
    # lines['Halpha'] = (8.7865, 8.7920)
    # lines['HeI6678'] = (8.8045, 8.8095)
    st = 'allFeII'

    st = 'allFeII'

    for line in lines.keys():
        print('finding', line, '...')
        with open(glob.glob(folder + '/in{}'.format(line))[0]) as f:
            dof += int(f.readlines()[-1])
    chisqfiles = list()
    for line in lines.keys():
        chisqfiles.append(glob.glob(folder + '/chisqs/chisq{}.npz'.format(line))[0])
    k1s, k2s, chisq = file_analyser(chisqfiles[0])

    for i in range(1, len(chisqfiles)):
        _, _, chisqhere = file_analyser(chisqfiles[i])
        chisq += chisqhere
    minima = get_minimum(k1s, k2s, chisq)
    idx = get_min_idx(chisq)
    print(minima)

    if len(uniques(k1s)) > 1:
        dof -= 2
        fig = plt.figure()
        ax = plot_contours(fig, k1s, k2s, chisq, dof)
        mark_minimum(ax, k1s, k2s, chisq, r"$\chi^2_{\textrm{red,min}}$")
        ax.legend(loc=2)
        ax.set_title(r"$\chi^2_{\textrm{red}}$")
        ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        ax.set_ylabel(r'$K_2(\si{\km\per\second})$')
        plt.grid()
        plt.tight_layout()
        fig.savefig(folder + '/chisq{}.png'.format(st), dpi=200)
        plt.close(fig)

        fig = plt.figure()
        ax = plot_uncertainty(fig, k1s, k2s, chisq, dof)
        ax.set_title(r'$1-P(\nu/2, \chi^2/2)$')
        ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        ax.set_ylabel(r'$K_2(\si{\km\per\second})$')
        plt.tight_layout()
        fig.savefig(folder + '/error{}.png'.format(st), dpi=200)
        plt.close(fig)

        fig = plt.figure()
        ax = plot_uncertainty_unscaled(fig, k1s, k2s, chisq, dof)
        ax.set_title(r'$1-P(\nu/2, \chi^2/2)$')
        ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        ax.set_ylabel(r'$K_2(\si{\km\per\second})$')
        plt.tight_layout()
        fig.savefig(folder + '/errorunscaled{}.png'.format(st), dpi=200)
        plt.close(fig)

        fig = plt.figure()
        ax = plot_oneDee(fig, k1s, chisq[:, idx[1]], dof)
        ax.set_title(r"$\chi^2_{\textrm{red}}, K_2 = $" + " " + str(k2s[idx[1]]) + " " + r'$\si{\km\per\second}$')
        ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        plt.grid()
        plt.tight_layout()
        fig.savefig(folder + '/chisqmargk2{}.png'.format(st), dpi=200)
        plt.close(fig)
    else:
        dof -= 1
    fig = plt.figure()
    ax = plot_oneDee(fig, k2s, chisq[idx[0], :], dof)
    ax.set_title(r"$\chi^2_{\textrm{red}}, K_1 = $" + " " + str(k1s[idx[0]]) + " " + r'$\si{\km\per\second}$')
    ax.set_xlabel(r'$K_2(\si{\km\per\second})$')
    plt.grid()
    plt.tight_layout()
    fig.savefig(folder + '/chisqmargk1{}.png'.format(st), dpi=200)
    plt.close(fig)

