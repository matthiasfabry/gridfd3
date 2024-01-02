"""
User script showing the typical use case for broad range fd3 separation.
First a list of files containing the relevant spectra is globbed.
Next, we specify the orbital parameters and the RV semi-amplitudes and various other
parameters such as the sampling rate and light factors
"""
import glob
import pathlib
import time

import modules.gridfd3classes as fd3classes

# input
# define working directories
obj = 'HR_6819/test'
fd3folder = obj

# find locations of your spectra
spectra_set = ['HR_6819/FEROS'
               ]  # allows for subsetting spectra
try:
    spec_folder = list()
    for folder in spectra_set:
        spec_folder.append(glob.glob('/Users/matthiasf/data/spectra/' +
                                     folder)[0])  # actual path to your folder containing spectra
except IndexError:
    spec_folder = None
    print('no spectra folder of object found')
    exit()

# geometrical orbit elements and its error. error is ignored if not monte_carlo
orbit = (40.335, 53116.9, 0, 270)  # p, t0, e, omega(A)
k1 = 60.4
k2 = 4.0

# lightfactors of your components (if thirdlight, give three)
lfs = [0.45, 0.55]
# do you want a (static) third component to be found?
thirdlight = False

# sampling of your spectra in angstrom
sampling = 0.2

# enter wavelength range(s) in natural log of wavelength and give name of line. Must be a dict.
lines = dict()
lines['range'] = (4010, 5000)
############################################################

starttime = time.time()
print('starting setup...')
print('orbit is:', orbit)
print('lightfactors are:', lfs)
print('spectroscopy folder is {}\n'.format(spec_folder))

# all fits files in this directory
allfiles = list()
for folder in spec_folder:
    allfiles.extend(glob.glob(folder + '/**/*.fits', recursive=True))

if len(allfiles) == 0:
    print('no spectra found')
    exit()
K = len(lines)
if K == 0:
    print('no lines selected')
    exit()

pathlib.Path(fd3folder).mkdir(parents=True, exist_ok=True)

# save the run_fd3 parameters for later reference
with open(fd3folder + "/params.txt", 'w') as paramfile:
    paramfile.write('orbit\t' + str(orbit) + '\n')
    paramfile.write('lightfactors\t' + str(lfs) + '\n')
    paramfile.write('sampling\t' + str(sampling) + '\n')
    paramfile.write('spectra\t' + str(spectra_set) + '\n')

fd3lineobjects = list()
print('building fd3gridline object for:')
for line in lines.keys():
    print(' {}'.format(line))
    fd3lineobjects.append(
        fd3classes.Fd3class(line, lines[line], sampling, allfiles, thirdlight, orbit, lfs=lfs,
                            k1=k1, k2=k2))

d3threads = list()
setuptime = time.time()
print('setup took {}s\n'.format(setuptime - starttime))
print('starting runs!')
now = time.time()
for line in fd3lineobjects:
    line.run_fd3(fd3folder)

print('Thanks for your patience! You waited a whopping {} hours!'.format(
    (time.time() - starttime) / 3600))
