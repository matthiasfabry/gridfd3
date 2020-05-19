import glob
import os
import pathlib
import time
from datetime import datetime

import numpy as np

import modules.gridfd3classes as fd3classes

# %%
starttime = time.time()

######
# input
fd3folder = 'HR_6819' + '/' + str(datetime.today().date()) + '_' + str(
    datetime.today().time().hour) + str(datetime.today().time().minute)
spectra_set = 'HR_6819/FEROS'
monte_carlo = False
N = 1000
k1str = '50 75 0.5'
k2str = '0 15 0.5'
orbit = (40.0366, 58889.1, 0, 270)  # p, t0, e, Omega(A)
orbit_err = (0.1, 0, 0.0, 0)

perturb_orbit = True
perturb_spectra = True
thirdlight = False
lfs = [0.5, 0.5]

# enter ln(lambda/A) range and name of line
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
lines['FeII4584'] = (8.4292, 8.4316)
# lines['HeII4686'] = (8.4510, 8.4535)
# lines['HeI4713'] = (8.4575, 8.4590)
# lines['Hbeta'] = (8.4850, 8.4925)
# lines['HeI5016'] = (8.5195, 8.5215)
lines['FeII5167'] = (8.5494, 8.5514)
lines['FeII5198'] = (8.5549, 8.5568)
lines['FeII5233'] = (8.5620, 8.5640)
lines['FeII5276'] = (8.5699, 8.5717)
lines['FeII5316'] = (8.5777, 8.5794)
# lines['HeII5411'] = (8.5935, 8.5990)
# lines['FeII5780'] = (8.6617, 8.6629)
# lines['CIV5801'] = (8.6652, 8.6667)
# lines['CIV5812'] = (8.6668, 8.6685)
# lines['HeI5875'] = (8.6775, 8.6795)
# lines['Halpha'] = (8.7865, 8.7920)
# lines['HeI6678'] = (8.8045, 8.8095)
sampling = 8e-6
######

print('starting setup...')

print('orbit is:', orbit)
print('lightfactors are:', lfs)

pathlib.Path(fd3folder).mkdir(parents=True, exist_ok=True)
spec_folder = None
try:
    spec_folder = glob.glob('/Users/matthiasf/data/spectra/' + spectra_set)[0]
except IndexError:
    print('no spectra folder of object found')
    exit()

print('spectroscopy folder is {}\n'.format(spec_folder))

# all fits files in this directory
allfiles = glob.glob(spec_folder + '/**/*.fits', recursive=True)
number_of_files = len(allfiles)

if number_of_files == 0:
    print('no spectra found')
    exit()

with open(fd3folder + "/params.txt", 'w') as paramfile:
    paramfile.write(str(orbit) + '\n')
    paramfile.write(str(orbit_err) + '\n')
    paramfile.write(str(lfs) + '\n')
    paramfile.write(str(sampling))
# noinspection PyTypeChecker
np.savetxt(fd3folder + '/orbit.txt', np.array(orbit))
# noinspection PyTypeChecker
np.savetxt(fd3folder + '/orbiterr.txt', np.array(orbit_err))
# noinspection PyTypeChecker
np.savetxt(fd3folder + '/lightfactors.txt', np.array(lfs))

K = len(lines)
if K == 0:
    print('no lines selected')
    exit()

cpus = os.cpu_count()
fd3lines = list()
print('building fd3gridline object for:')
for line in lines.keys():
    print(' {}'.format(line))
    fd3lines.append(
        fd3classes.Fd3gridLine(line, lines[line][0:2], sampling, allfiles, monte_carlo, thirdlight, lfs, orbit,
                               orbit_err, perturb_orbit, perturb_spectra, k1str, k2str))

threads = list()
if monte_carlo:
    # create threads
    print('number of threads will be {}'.format(cpus))
    print('each thread will have {} iterations to complete'.format(N / cpus))
    atleast = int(N / cpus)
    remainder = int(N % cpus)

    for i in range(remainder):
        threads.append(fd3classes.Fd3gridMCThread(fd3folder, i + 1, atleast + 1, fd3lines))
    for i in range(remainder, cpus):
        threads.append(fd3classes.Fd3gridMCThread(fd3folder, i + 1, atleast, fd3lines))

else:
    for fd3line in fd3lines:
        threads.append(fd3classes.Fd3gridThread(fd3folder, fd3line))

setuptime = time.time()
print('setup took {}s\n'.format(setuptime - starttime))

print('starting runs!')
for thread in threads:
    thread.start()

for thread in threads:
    thread.join()

print('All runs done in {}h, see you later!'.format((time.time() - starttime) / 3600))
