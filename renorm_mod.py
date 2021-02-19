#%%
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.interpolate as spint
folder = '9_Sgr/fd3'
#%%
lfs = [0.62, 0.38]
thresh = 0.002
smooth = 0.02
name = 'range'


x = np.loadtxt('{}/products{}.mod'.format(folder, name)).T
x[0] = np.exp(x[0])
fig, ax = plt.subplots(ncols=2, sharex='all')

x[1] *= lfs[0]
x[2] *= lfs[1]

# plt.show()
# print(x[0], x[1])
inds = abs(x[1] + x[2] - 1) < thresh

ax[1].plot(x[0], x[1] + x[2] - 1, 'g', label='sum')
ax[1].plot(x[0][inds], x[1][inds]+x[2][inds] - 1, 'ko', label='selection')

# plt.plot(x[0][inds], x[1][inds], 'bo')
# plt.plot(x[0][inds], x[2][inds], 'ro')
x_avg = (x[1][inds] + 1 - x[2][inds]) / 2
spline = spint.UnivariateSpline(x[0][inds], x_avg, s=smooth)
ax[1].plot(x[0][inds], x_avg, label='avgs')
ax[1].plot(x[0], spline(x[0]), label='spline eval')
ax[1].legend()
# plt.show()
x1 = x[1] - spline(x[0])
x2 = x[2] - 1 + spline(x[0])
prim = np.array([x[0], x1 / lfs[0] + 1]).T
sec = np.array([x[0], x2 / lfs[1] + 1]).T
ax[0].plot(x[0], x1 / lfs[0] + 1 + 0.1, 'b', label='prim + 0.1')
ax[0].plot(x[0], x2 / lfs[1] + 1, 'r', label='sec')
ax[0].legend()
fig.tight_layout()
plt.show()
errorp = np.std(prim[:20, 1])
errors = np.std(sec[:20, 1])
#%%
np.savetxt(folder + '/{}primary.txt'.format(name), np.array([x[0], x[1], errorp * np.ones(len(x[1]))]).T)
np.savetxt(folder + '/{}secondary.txt'.format(name), np.array([x[0], x[2], errors * np.ones(len(x[2]))]).T)
np.savetxt(folder + '/{}primary_norm.txt'.format(name),
       np.array([x[0], x1 / lfs[0] + 1, errorp * np.ones(len(x[1]))]).T)
np.savetxt(folder + '/{}secondary_norm.txt'.format(name),
       np.array([x[0], x2 / lfs[1] + 1, errors * np.ones(len(x[2]))]).T)