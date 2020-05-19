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


lines = dict()
# lines['Heta'] = (8.2490, 8.2545, 1e-5)
# lines['Hzeta'] = (8.2630, 8.2685, 1e-5)
# lines['FeI3923'] = (8.2730, 8.2765, 1e-5)
# lines['Hepsilon'] = (8.2845, 8.2888)
lines['HeI+II4026'] = (8.2990, 8.302, 5e-6)
# lines['NIV4058'] = (8.3080, 8.3088, 1e-5)
# lines['Hdelta'] = (8.3160, 8.3215, 1e-5)
# lines['SiIV4116'] = (8.3215, 8.3238, 1e-5)
lines['HeI4143'] = (8.3280, 8.3305, 5e-6)
lines['HeII4200'] = (8.3412, 8.3444, 5e-6)
# lines['Hgamma'] = (8.3730, 8.3785, 1e-5)
# lines['NIII4379'] = (8.3835, 8.3857, 1e-5)
lines['HeI4387'] = (8.3855, 8.3875, 5e-6)
lines['HeI4471'] = (8.4047, 8.4064, 5e-6)
lines['HeII4541'] = (8.4195, 8.4226, 5e-6)
# lines['NV4604+4620'] = (8.4338, 8.4390, 1e-5)
lines['HeII4686'] = (8.4510, 8.4534, 5e-6)
lines['HeI4713'] = (8.4577, 8.4585, 5e-6)
# lines['Hbeta'] = (8.4860, 8.4920, 1e-5)
lines['HeI4922'] = (8.5012, 8.5017, 5e-6)
lines['HeI5016'] = (8.5200, 8.5210, 5e-6)
lines['HeII5411'] = (8.5940, 8.5986, 5e-6)
# lines['OIII5592'] = (8.6281, 8.6300, 1e-5)
# lines['CIII5696'] = (8.6466, 8.6482, 1e-5)
# lines['FeII5780'] = (8.6617, 8.6629, 1e-5)
# lines['CIV5801'] = (8.6652, 8.6667, 1e-5)
# lines['CIV5812'] = (8.6668, 8.6685, 1e-5)
lines['HeI5875'] = (8.6777, 8.6794, 5e-6)
# lines['Halpha'] = (8.7865, 8.7920, 1e-5)
lines['HeI6678'] = (8.805, 8.8095, 5e-6)

files = list()
mink1s, mink2s = list(), list()
k1s, k2s = None, None
N = 256
threads = 16
atleast = int(N / threads)
remainder = int(N % threads)

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
                chisqit = np.zeros((len(k1s), len(k2s)))
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
