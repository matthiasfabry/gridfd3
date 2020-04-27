import numpy as np
import matplotlib.pyplot as plt
import outfile_analyser as oa
import matplotlib as mpl
mpl.use('pgf')
pgf_with_latex = {                      # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
    "text.usetex": True,                # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],                   # blank entries should cause plots
    "font.sans-serif": [],              # to inherit fonts from the document
    "font.monospace": [],
    "axes.labelsize": 16,               # LaTeX default is 10pt font.
    "font.size": 16,
    "legend.fontsize": 8,               # Make the legend/label fonts
    "xtick.labelsize": 12,               # a little smaller
    "ytick.labelsize": 12,
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts
        r"\usepackage[T1]{fontenc}",        # plots will be generated
        r"\usepackage{siunitx}"
        ]                                   # using this preamble
    }

plt.rcParams.update(pgf_with_latex)
folders = ['LB-1/HERMES', 'LB-1/HERMES/lb1letter', 'LB-1/HERMES/lb1letter2']

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
plt.figure()
plt.hist(mink1s, bins=oa.uniques(mink1s), align='left', color='r')
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