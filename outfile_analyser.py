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
    contour = aax.contourf(k1grid, k2grid, cchisq / ddof, levels=50, cmap='inferno')
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

    folder = '9_Sgr'
    dof = 0
    lines = dict()
    # lines['Heta'] = (8.2490, 8.2545, 1e-5)
    # lines['Hzeta'] = (8.2630, 8.2685, 1e-5)
    # lines['FeI3923'] = (8.2730, 8.2765, 1e-5)
    # lines['Hepsilon'] = (8.2845, 8.2888)
    # lines['HeI+II4026'] = (8.2990, 8.302, 1e-5)
    # lines['NIV4058'] = (8.3080, 8.3088, 1e-5)
    lines['Hdelta'] = (8.3160, 8.3215, 1e-5)
    # lines['SiIV4116'] = (8.3215, 8.3238, 1e-5)
    # lines['HeI4143'] = (8.3280, 8.3305, 1e-5)
    lines['HeII4200'] = (8.3412, 8.3444, 1e-5)
    lines['Hgamma'] = (8.3730, 8.3785, 1e-5)
    # lines['NIII4379'] = (8.3835, 8.3857, 1e-5)
    # lines['HeI4387'] = (8.3855, 8.3875, 1e-5)
    # lines['HeI4471'] = (8.4047, 8.4064, 1e-5)
    lines['HeII4541'] = (8.4195, 8.4226, 1e-5)
    lines['NV4604+4620'] = (8.4338, 8.4390, 1e-5)
    lines['HeII4686'] = (8.4510, 8.4534, 1e-5)
    # lines['HeI4713'] = (8.4577, 8.4585, 1e-5)
    lines['Hbeta'] = (8.4860, 8.4920, 1e-5)
    # lines['HeI4922'] = (8.5012, 8.5017, 1e-5)
    # lines['HeI5016'] = (8.5200, 8.5210, 1e-5)
    lines['HeII5411'] = (8.5940, 8.5986, 1e-5)
    # lines['OIII5592'] = (8.6281, 8.6300, 1e-5)
    # lines['CIII5696'] = (8.6466, 8.6482, 1e-5)
    # lines['FeII5780'] = (8.6617, 8.6629, 1e-5)
    # lines['CIV5801'] = (8.6652, 8.6667, 1e-5)
    # lines['CIV5812'] = (8.6668, 8.6685, 1e-5)
    # lines['HeI5875'] = (8.6777, 8.6794, 1e-5)
    # lines['Halpha'] = (8.7865, 8.7920, 1e-5)
    # lines['HeI6678'] = (8.805, 8.8095, 1e-5)

    for line in lines.keys():
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
        ax.set_title(r"$\chi^2_{\textrm{red}}$")
        ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        ax.set_ylabel(r'$K_2(\si{\km\per\second})$')
        mark_minimum(ax, k1s, k2s, chisq, r"$\chi^2_{\textrm{red,min}}$")
        ax.legend(loc=2)
        plt.tight_layout()
        fig.savefig(folder + '/chisq{}.png'.format('allG2'), dpi=200)

        fig = plt.figure()
        ax = plot_uncertainty(fig, k1s, k2s, chisq, dof)
        fig.savefig(folder + '/errorallG2.png', dpi=200)
        #
        # fig = plt.figure()
        # ax = plot_oneDee(fig, k1s, chisq[:, idx[1]], dof)
        # ax.set_title(r"$\chi^2_{\textrm{red}}, K_2 = $" + " " + str(k2s[idx[1]]) + " " + r'$\si{\km\per\second}$')
        # ax.set_xlabel(r'$K_1(\si{\km\per\second})$')
        # plt.grid()
        # plt.tight_layout()
        # fig.savefig(folder + '/chisqmargk2.png', dpi=200)
    # else:
    #     dof -= 1
    # fig = plt.figure()
    # ax = plot_oneDee(fig, k2s, chisq[idx[0], :], dof)
    # ax.set_title(r"$\chi^2_{\textrm{red}}, K_1 = $" + " " + str(k1s[idx[0]]) + " " + r'$\si{\km\per\second}$')
    # ax.set_xlabel(r'$K_2(\si{\km\per\second})$')
    # plt.grid()
    # plt.tight_layout()
    # fig.savefig(folder + '/chisqmargk1.png', dpi=200)
