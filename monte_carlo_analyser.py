import glob
import sys
import os
import numpy as np
import outfile_analyser as oa
# noinspection PyUnresolvedReferences
import plotsetup

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
N = 3000
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
np.save(folder+'/mink1s', mink1s)
np.save(folder+'/mink2s', mink2s)
