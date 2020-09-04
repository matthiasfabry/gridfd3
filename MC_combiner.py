# %%
# noinspection PyUnresolvedReferences
import plotsetup
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import scipy.stats as stat

import modules.orbitalfunctions as of


folders = ['9_Sgr/first', '9_Sgr/second', '9_Sgr/third']

mink1sog = None
mink2sog = None
combsog = None
for folder in folders:
    mink1here = np.load(folder + '/mink1s.npy')
    mink2here = np.load(folder + '/mink2s.npy')
    combshere = np.load(folder + '/combs.npy')
    if mink1sog is None:
        mink1sog = mink1here
    else:
        mink1sog = np.concatenate((mink1sog, mink1here))
    if mink2sog is None:
        mink2sog = mink2here
    else:
        mink2sog = np.concatenate((mink2sog, mink2here))
    if combsog is None:
        combsog = combshere
    else:
        combsog = np.concatenate((combsog, combshere))
print(len(combsog), len(mink1sog), len(mink2sog))


def populate_mass_line(m):
    lst = list()
    for k1 in uk1s:
        def lower(k2):
            return m - of.total_mass(0.648, k1, k2, 3261, np.sin(86.5 * np.pi / 180))

        lst.append((k1, spopt.root_scalar(lower, x0=1, method='toms748', bracket=(0, 100)).root))
    return np.array(lst)


# %%
# non trimmed
N = len(mink1sog)
k1q1 = mink1sog[int(0.158 * N)]
k1q2 = mink1sog[int(0.842 * N)]
k2q1 = mink2sog[int(0.158 * N)]
k2q2 = mink2sog[int(0.842 * N)]
uk1s = np.unique(mink1sog)
uk2s = np.unique(mink2sog)
ucombs, num = np.unique(combsog, axis=0, return_counts=True)
lowers = populate_mass_line(80.87)
uppers = populate_mass_line(88.27)
mids = populate_mass_line(84.57)
threeuppers = populate_mass_line(84.57 + 3 * 3.7)
threelowers = populate_mass_line(84.57 - 3 * 3.7)
print(min(mink1sog), max(mink1sog))
histk1 = np.histogram(mink1sog, bins=range(int(min(mink1sog)), int(max(mink1sog) + 2)))
histk2 = np.histogram(mink2sog, bins=range(int(min(mink2sog)), int(max(mink2sog) + 2)))
print(histk1[0])
print(histk1[1])
# if len(uk1s) != 1:
#     plt.figure()
#     plt.hist(np.round(mink1sog, 2), bins=uk1s, align='left', color='b')
#     plt.xlabel(r'$K_1 (\si{\km\per\second})$')
#     plt.ylabel(r'$N$')
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig('histk1.png', dpi=200)
#     plt.close()
#     plt.figure()
#     plt.hist(np.round(mink1sog, 2), bins=uk1s, align='left', color='b', label=r'$K_1$')
#     plt.hist(np.round(mink2sog, 2), bins=uk2s, align='left', color='r', label=r'$K_2$')
#     plt.grid()
#     plt.legend()
#     plt.xlabel(r'$K (\si{\km\per\second})$')
#     plt.ylabel(r'$N$')
#     plt.tight_layout()
#     plt.savefig('histk1k2.png', dpi=200)
#     plt.close()
#     plt.figure()
#     plt.scatter(ucombs[:, 0], ucombs[:, 1], c=num, cmap='inferno', lw=0, ms=2)
#     plt.plot(threelowers[:, 0], threelowers[:, 1], 'b-.', lw=0.5)
#     plt.plot(lowers[:, 0], lowers[:, 1], 'b--', lw=0.5)
#     plt.plot(mids[:, 0], mids[:, 1], 'b', lw=1)
#     plt.plot(uppers[:, 0], uppers[:, 1], 'b--', lw=0.5)
#     plt.plot(threeuppers[:, 0], threeuppers[:, 1], 'b-.', lw=0.5)
#     plt.xlim((14, 44))
#     plt.ylim((45, 65))
#     plt.colorbar()
#     plt.xlabel(r'$K_1 (\si{\km\per\second})$')
#     plt.ylabel(r'$K_2 (\si{\km\per\second})$')
#     plt.savefig('k2vsk2.png', dpi=200)
#     plt.close()
# plt.figure()
# plt.hist(np.round(mink2sog, 2), bins=uk2s, align='left', color='r')
# plt.xlabel(r'$K_2 (\si{\km\per\second})$')
# plt.ylabel(r'$N$')
# plt.grid()
# plt.tight_layout()
# plt.savefig('histk2.png', dpi=200)
# plt.close()
print(np.average(mink1sog), np.average(mink2sog))
print(stat.mode(mink1sog), stat.mode(mink2sog))
print(np.std(mink1sog), np.std(mink2sog))
print('k1 1sig =', k1q1, k1q2)
print('k2 1sig =', k2q1, k2q2)

# %%
# trimmed run!
trimmedcombs = np.copy(combsog)
trimmedcombs = trimmedcombs[
    of.total_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261, np.sin(86.5 * np.pi / 180)) < 84.57 + 3 * 3.7]
trimmedcombs = trimmedcombs[
    of.total_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261, np.sin(86.5 * np.pi / 180)) > 84.57 - 3 * 3.7]
Ntrim = len(trimmedcombs)
print(Ntrim, 'samples after trim')

k1strim = sorted(trimmedcombs[:, 0])
k2strim = sorted(trimmedcombs[:, 1])
uk1strim = np.unique(k1strim)
uk2strim = np.unique(k2strim)
ucombstrim, numtrim = np.unique(trimmedcombs, axis=0, return_counts=True)

if len(uk1s) != 1:
    plt.figure()
    plt.hist(np.round(k1strim, 2), bins=uk1strim, align='left', color='b')
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$N$')
    plt.grid()
    plt.tight_layout()
    plt.savefig('histk1trim.png', dpi=200)
    plt.close()
    plt.figure()
    plt.bar(histk1[1][:-1], histk1[0], width=1, fill=False, edgecolor='b', label=r'$K_1\,{\rm rejected}$', hatch='///')
    plt.bar(histk2[1][:-1], histk2[0], width=1, fill=False, edgecolor='r', label=r'$K_2\,{\rm rejected}$', hatch='///')
    plt.hist(np.round(k1strim, 2), bins=uk1strim, align='left', color='b', label=r'$K_1$')
    plt.hist(np.round(k2strim, 2), bins=uk2strim, align='left', color='r', label=r'$K_2$')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$K (\si{\km\per\second})$')
    plt.ylabel(r'$N$')
    plt.tight_layout()
    plt.savefig('histk1k2trim.png', dpi=200)
    plt.close()
    plt.figure()
    plt.scatter(ucombs[:, 0], ucombs[:, 1], c=num, cmap='inferno', lw=0, s=30, vmin=1, vmax=max(num), marker='^')
    plt.scatter(ucombstrim[:, 0], ucombstrim[:, 1], c=numtrim, cmap='inferno', lw=0, s=60, vmin=1, vmax=max(num))
    plt.plot(threelowers[:, 0], threelowers[:, 1], 'b-.', lw=0.5)
    plt.plot(lowers[:, 0], lowers[:, 1], 'b--', lw=0.5)
    plt.plot(mids[:, 0], mids[:, 1], 'b', lw=1)
    plt.plot(uppers[:, 0], uppers[:, 1], 'b--', lw=0.5)
    plt.plot(threeuppers[:, 0], threeuppers[:, 1], 'b-.', lw=0.5)
    plt.ylim((45, 65))
    plt.colorbar()
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$K_2 (\si{\km\per\second})$')
    plt.xlim((14, 44))
    plt.ylim((45, 65))
    plt.savefig('k2vsk2trim.png', dpi=200)
    plt.close()
plt.figure()
plt.hist(np.round(k2strim, 2), bins=uk2strim, align='left', color='r')
plt.xlabel(r'$K_2 (\si{\km\per\second})$')
plt.ylabel(r'$N$')
plt.grid()
plt.tight_layout()
plt.savefig('histk2trim.png', dpi=200)
plt.close()

k1q1 = k1strim[int(0.158 * Ntrim)]
k1q2 = k1strim[int(0.842 * Ntrim)]
k2q1 = k2strim[int(0.158 * Ntrim)]
k2q2 = k2strim[int(0.842 * Ntrim)]
print(np.average(k1strim), np.average(k2strim))
print(stat.mode(k1strim), stat.mode(k2strim))
print(np.std(k1strim), np.std(k2strim))
print('k1trim 1sig =', k1q1, k1q2)
print('k2trim 1sig =', k2q1, k2q2)
