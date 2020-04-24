import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
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

folder = None
try:
    folder = sys.argv[1]
except IndexError:
    print("Give a folder to work on!")
    exit()


lines = set()
outfiles = glob.glob(folder + '/thread1/out*')

for file in outfiles:
    lines.add(file.split('out')[-1])


files = list()
mink1s, mink2s = list(), list()
k1s, k2s = None, None
N = 1000
threads = os.cpu_count()
atleast = int(N / threads)
remainder = int(N % threads)

filename = 'LB-1'
for c in range(threads):
    if c < remainder:
        iterations = atleast + 1
    else:
        iterations = atleast
    for i in range(iterations):
        k1it, k2it, chisqit = None, None, None
        for line in lines:
            files = glob.glob(folder + '/thread{}/chisqs/chisq{}{}.npz'.format(c + 1, line, i + 1), recursive=True)
            if files is None or len(files) != 1:
                print('either no file or too many files! at thread', c + 1, 'line', line, 'iteration', i + 1)
                exit()

            k1shere, k2shere, chisqhere = oa.file_analyser(files[0])
            if k1it is None:
                k1s = k1shere
            if k2it is None:
                k2s = k2shere
            if chisqit is None:
                chisqit = np.zeros((len(k1s), len(k2s))).T
            chisqit += chisqhere

        mink1, mink2 = oa.get_minimum(k1s, k2s, chisqit)
        mink1s.append(mink1)
        mink2s.append(mink2)


if k1s is None or k2s is None:
    print('nothing to report')
    exit()

mink1s = sorted(mink1s)
mink2s = sorted(mink2s)

k1q1 = mink1s[int(0.025*N)]
k1q2 = mink1s[int(0.975*N)]
k2q1 = mink2s[int(0.025*N)]
k2q2 = mink2s[int(0.975*N)]

plt.figure()
plt.hist(mink1s, bins=len(range(int(min(mink1s)), int(max(mink1s)))), align='left', color='r', label=r'$K_1$')
plt.hist(mink2s, bins=len(range(int(min(mink2s)), int(max(mink2s)))), align='left', color='b', label=r'$K_2$')
plt.xlabel(r'$K (\si{\km\per\second})$')
plt.ylabel(r'$N$')
plt.legend()
plt.tight_layout()
plt.savefig(folder + '/hist{}.png'.format(filename), dpi=200)

print(np.average(mink1s), np.average(mink2s))
print(np.std(mink1s), np.std(mink2s))
print('k1 95% =', k1q1, k1q2)
print('k2 95% =', k2q1, k2q2)
