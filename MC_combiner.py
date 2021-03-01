# %%
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spopt
import scipy.stats as stat
import plotsetup  # noqa
import modules.orbitalfunctions as of

folders = ['9_Sgr/recombMCcovarresub2',
           # '9_Sgr/recombMC2'
           ]

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


def multi_median(ms):
    x = [point[0] for point in ms]
    y = [point[1] for point in ms]
    
    x0 = np.array([sum(x) / len(x), sum(y) / len(y)])
    
    def dist_func(x0):
        return sum(
            ((np.full(len(x), x0[0]) - x) ** 2 + (np.full(len(x), x0[1]) - y) ** 2) ** (1 / 2))
    
    res = spopt.minimize(dist_func, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})
    return res.x


def populate_mass_line(m):
    lst = list()
    for k1 in uk1s:
        def lower(k2):
            return m - of.total_mass(0.648, k1, k2, 3261, np.sin(86.5 * np.pi / 180))
        
        lst.append((k1, spopt.root_scalar(lower, x0=1, method='toms748', bracket=(0, 100)).root))
    return np.array(lst)


def stats(lst, name):
    lst = np.sort(lst)
    NN = len(lst)
    q1 = lst[int(0.158 * NN)]
    q2 = lst[int(0.842 * NN)]
    med = lst[int(0.5 * NN)]
    print(name, 'avg', np.average(lst))
    print(name, 'median', med)
    print(name, 'upper 1sigma', q2 - med)
    print(name, 'lower 1sigma', med - q1)
    print(name, 'std', np.std(lst))
    print(name, 'mode', stat.mode(lst))


# %%
# non trimmed
limiter = 1000
mink1slim = mink1sog[:limiter]
mink2slim = mink2sog[:limiter]
combslim = combsog[:limiter]

print(N := len(mink1slim))

#%%
k1q1 = mink1slim[int(0.158 * N)]
k1q2 = mink1slim[int(0.842 * N)]
k2q1 = mink2slim[int(0.158 * N)]
k2q2 = mink2slim[int(0.842 * N)]
uk1s = np.unique(mink1slim)
uk2s = np.unique(mink2slim)
ucombs, num = np.unique(combslim, axis=0, return_counts=True)
# lowers = populate_mass_line(84.87)
# uppers = populate_mass_line(92.27)
# mids = populate_mass_line(88.57)
# threeuppers = populate_mass_line(88.57 + 3 * 3.7)
# threelowers = populate_mass_line(88.57 - 3 * 3.7)

m1s = of.primary_mass(0.648, combslim[:, 0], combslim[:, 1], 3261, np.sin(86.5 * np.pi / 180))
m2s = of.secondary_mass(0.648, combslim[:, 0], combslim[:, 1], 3261, np.sin(86.5 * np.pi / 180))
mcombs = np.array([m1s, m2s]).T
um1s = np.unique(m1s)
um2s = np.unique(m2s)
mucombs, munum = np.unique(mcombs, axis=0, return_counts=True)

if len(uk1s) != 1:
    plt.figure()
    plt.hist(np.round(mink1slim, 2), bins=uk1s, align='left', color='b')
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$N$')
    plt.grid()
    plt.tight_layout()
    plt.savefig(folders[0] + '/histk1.png', dpi=200)
    plt.close()
    plt.figure()
    plt.hist(np.round(mink1slim, 2), bins=uk1s, align='left', color='b', label=r'$K_1$')
    plt.hist(np.round(mink2slim, 2), bins=uk2s, align='left', color='r', label=r'$K_2$')
    plt.grid()
    plt.legend()
    plt.xlabel(r'$K (\si{\km\per\second})$')
    plt.ylabel(r'$N$')
    plt.tight_layout()
    plt.savefig(folders[0] + '/histk1k2.png', dpi=200)
    plt.close()
    plt.figure()
    plt.grid()
    
    plt.scatter(ucombs[:, 0], ucombs[:, 1], c=num, cmap='inferno', lw=0, s=50)
    plt.colorbar(label=r'$N$')
    plt.scatter([36], [49], edgecolors='red', facecolors='none', lw=2, s=80, label=r'Original Data')
    plt.xlim((22, 45))
    plt.ylim((42, 59))
    plt.legend()
    
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$K_2 (\si{\km\per\second})$')
    plt.savefig(folders[0] + '/k1vsk2.png', dpi=200)
    plt.close()
    plt.figure()
    plt.grid()
    plt.scatter(mucombs[:, 0], mucombs[:, 1], c=munum, cmap='inferno', lw=0, s=50)
    plt.colorbar(label=r'$N$')
    
    plt.xlabel(r'$M_1 ({\rm M}_\odot)$')
    plt.ylabel(r'$M_2 ({\rm M}_\odot)$')
    plt.savefig(folders[0] + '/m1vsm2.png', dpi=200)
    plt.close()
plt.figure()
plt.hist(np.round(mink2slim, 2), bins=uk2s, align='left', color='r')
plt.xlabel(r'$K_2 (\si{\km\per\second})$')
plt.ylabel(r'$N$')
plt.grid()
plt.tight_layout()
plt.savefig(folders[0] + '/histk2.png', dpi=200)
plt.close()
print(multi_median(combslim))
print(multi_median(mcombs))
stats(mink1slim, 'k1')
stats(mink2slim, 'k2')
stats(m1s, 'M1')
stats(m2s, 'M2')

# %%
# trimmed run_fd3!
trimmedcombs = np.copy(combsog)
trimmedcombs = trimmedcombs[
    of.total_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261,
                  np.sin(86.5 * np.pi / 180)) < 93.84 + 3 * 3.7]
trimmedcombs = trimmedcombs[
    of.total_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261,
                  np.sin(86.5 * np.pi / 180)) > 93.84 - 3 * 3.7]
Ntrim = len(trimmedcombs)
print(Ntrim, 'samples after trim')

k1strim = sorted(trimmedcombs[:, 0])
k2strim = sorted(trimmedcombs[:, 1])
uk1strim = np.unique(k1strim)
uk2strim = np.unique(k2strim)
ucombstrim, numtrim = np.unique(trimmedcombs, axis=0, return_counts=True)

if len(uk1s) != 1:
    # plt.figure()
    # plt.hist(np.round(k1strim, 2), bins=uk1strim, align='left', color='b')
    # plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    # plt.ylabel(r'$N$')
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(folders[0]+'/histk1trim.png', dpi=200)
    # plt.close()
    # plt.figure()
    #
    # plt.hist(k2strim, bins=uk2strim, align='left', color='r', label=r'$K_2$')
    # plt.hist(k1strim, bins=uk1strim, align='left', color='b', label=r'$K_1$')
    # plt.bar(histk1[1][:-1], histk1[0], width=1, fill=False, edgecolor='b', label=r'$K_1\,
    # {\rm rejected}$', hatch='///')
    # plt.bar(histk2[1][:-1], histk2[0], width=1, fill=False, edgecolor='r', label=r'$K_2\,
    # {\rm rejected}$', hatch='///')
    # plt.grid()
    # plt.legend()
    # plt.xlabel(r'$K (\si{\km\per\second})$')
    # plt.ylabel(r'$N$')
    # plt.tight_layout()
    # plt.savefig(folders[0]+'/histk1k2trim.png', dpi=200)
    # plt.close()
    plt.figure()
    plt.scatter(ucombs[:, 0], ucombs[:, 1], c=num, cmap='inferno', lw=0, s=30, vmin=1,
                vmax=max(num), marker='^')
    plt.scatter(ucombstrim[:, 0], ucombstrim[:, 1], c=numtrim, cmap='inferno', lw=0, s=60, vmin=1,
                vmax=max(num))
    # plt.plot(threelowers[:, 0], threelowers[:, 1], 'b-.', lw=0.5)
    # plt.plot(lowers[:, 0], lowers[:, 1], 'b--', lw=0.5)
    # plt.plot(mids[:, 0], mids[:, 1], 'b', lw=1)
    # plt.plot(uppers[:, 0], uppers[:, 1], 'b--', lw=0.5)
    # plt.plot(threeuppers[:, 0], threeuppers[:, 1], 'b-.', lw=0.5)
    
    plt.ylim((45, 65))
    plt.colorbar()
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$K_2 (\si{\km\per\second})$')
    plt.xlim((14, 44))
    plt.ylim((45, 65))
    plt.savefig(folders[0] + '/k1vsk2trim.png', dpi=200)
    plt.close()
plt.figure()
plt.hist(np.round(k2strim, 2), bins=uk2strim, align='left', color='r')
plt.xlabel(r'$K_2 (\si{\km\per\second})$')
plt.ylabel(r'$N$')
plt.grid()
plt.tight_layout()
plt.savefig(folders[0] + '/histk2trim.png', dpi=200)
plt.close()

stats(k1strim, 'K1')
stats(k2strim, 'K2')

m1strim = of.primary_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261,
                          np.sin(86.5 * np.pi / 180))
m2strim = of.secondary_mass(0.648, trimmedcombs[:, 0], trimmedcombs[:, 1], 3261,
                            np.sin(86.5 * np.pi / 180))
mcombstrim = np.array([m1strim, m2strim]).T
um1strim = np.unique(np.round(m1strim))
um2strim = np.unique(np.round(m2strim))
mucombstrim, munumtrim = np.unique(np.round(mcombstrim), axis=0, return_counts=True)

plt.figure()
plt.hist(np.round(m1s, 2), bins=um1s, align='left', color='b')
plt.xlabel(r'$M_1$')
plt.ylabel(r'$N$')
plt.grid()
plt.tight_layout()
plt.savefig(folders[0] + '/histm1.png', dpi=200)
plt.close()
plt.figure()
plt.hist(np.round(m1s, 2), bins=um1s, align='left', color='b', label=r'$K_1$')
plt.hist(np.round(m2s, 2), bins=um2s, align='left', color='r', label=r'$K_2$')
plt.grid()
plt.legend()
plt.xlabel(r'$M$')
plt.ylabel(r'$N$')
plt.tight_layout()
plt.savefig(folders[0] + '/histm1m2.png', dpi=200)
plt.close()
plt.figure()
plt.scatter(mucombs[:, 0], mucombs[:, 1], c=munum, cmap='inferno', lw=0, s=30, vmin=1,
            vmax=max(munum), marker='^')
plt.scatter(mucombstrim[:, 0], mucombstrim[:, 1], c=munumtrim, cmap='inferno', lw=0, s=60, vmin=1,
            vmax=max(munum))
# plt.plot(threelowers[:, 0], threelowers[:, 1], 'b-.', lw=0.5)
# plt.plot(lowers[:, 0], lowers[:, 1], 'b--', lw=0.5)
# plt.plot(mids[:, 0], mids[:, 1], 'b', lw=1)
# plt.plot(uppers[:, 0], uppers[:, 1], 'b--', lw=0.5)
# plt.plot(threeuppers[:, 0], threeuppers[:, 1], 'b-.', lw=0.5)
plt.colorbar()
plt.xlabel(r'$M_1$')
plt.ylabel(r'$M_2$')
plt.savefig(folders[0] + '/m1vsm2trim.png', dpi=200)
plt.close()
stats(m1s, 'M1')
stats(m2s, 'M2')
