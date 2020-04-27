import numpy as np
import matplotlib.pyplot as plt
import outfile_analyser as oa
# noinspection PyUnresolvedReferences
import plotsetup

folders = ['LB-1/HERMES']

mink1s = np.array([])
mink2s = np.array([])
for folder in folders:
    mink1here = np.load(folder + '/mink1s.npy')
    mink2here = np.load(folder + '/mink2s.npy')
    mink1s = np.hstack((mink1s, mink1here))
    mink2s = np.hstack((mink2s, mink2here))


N = len(mink1s)
print(N)
mink1s = sorted(mink1s)
mink2s = sorted(mink2s)
k1q1 = mink1s[int(0.158*N)]
k1q2 = mink1s[int(0.842*N)]
k2q1 = mink2s[int(0.158*N)]
k2q2 = mink2s[int(0.842*N)]
uk1s = oa.uniques(mink1s)
if len(uk1s) != 1:
    plt.figure()
    plt.hist(np.round(mink1s, 2), bins=uk1s, align='left', color='r')
    plt.xlabel(r'$K_1 (\si{\km\per\second})$')
    plt.ylabel(r'$N$')
    plt.tight_layout()
    plt.savefig('histk1.png', dpi=200)
plt.figure()
plt.hist(mink2s, bins=oa.uniques(mink2s), align='left', color='b')
plt.xlabel(r'$K_2 (\si{\km\per\second})$')
plt.ylabel(r'$N$')
plt.tight_layout()
plt.savefig('histk2.png', dpi=200)

print(np.average(mink1s), np.average(mink2s))
print(np.std(mink1s), np.std(mink2s))
print('k1 1sig =', k1q1, k1q2)
print('k2 1sig =', k2q1, k2q2)