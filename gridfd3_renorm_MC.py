"""
User script showing the typical use case for monte carlo sampling the gridfd3 procedure.
First a list of files containing the relevant spectra is globbed.
Next, we specify the orbital parameters and the RV semi-amplitude ranges that need to be explored, and various other
parameters such as the sampling rate and light factors
"""
import glob
import pathlib
import time

import numpy as np

import modules.gridfd3classes as fd3classes

# input

# specify working directories, can be anything you want
obj = '9_Sgr'
gridfd3folder = obj + '/recombMCcovar'
fd3folder = gridfd3folder + '/fd3'
spectra_set = ['9_Sgr']  # listing allows for subsetting spectra

try:
    spec_folder = list()
    for folder in spectra_set:
        spec_folder.append(glob.glob('/Users/matthiasf/data/spectra/' +
                                     folder)[0])  # actual path to your folder containing spectra
except IndexError:
    spec_folder = None
    print('no spectra folder of object found')
    exit()

# K1 and K2 ranges to be explored, in string form: 'left right step', all in km/s
k1str = '21 41 1'
k2str = '45 65 1'

# Indicate if you want the model spectra to be computed
back = False

# geometrical orbit elements and its error. error is ignored if not monte_carlo
orbit = (3261, 56547, 0.648, 30.7)  # p, t0, e, omega(A)
orbit_err = np.array([[68, 12, 0.009, 2.3]])

# optional: correlation matrix of the parameters
orbit_err_unscale = np.array(
    [[14.2, 2.396, 0.002, 0.452]])  # not properly scaled errors corresponding to the matrix below
orbit_covar_unscale = np.array(
    [[2.016706010889813285e+02, 2.650853464144408811e+00, 1.653940436054600338e-02, 1.510087481644369234e+00],
     [2.650853464144408811e+00, 5.739323098131123402e+00, 3.250949346561587527e-03, 1.040789875201268533e+00],
     [1.653940436054600338e-02, 3.250949346561587527e-03, 3.299601439411175585e-06, 6.442714780292257666e-04],
     [1.510087481644369234e+00, 1.040789875201268533e+00, 6.442714780292257666e-04,
      2.043357582025700225e-01]])  # correlation matrix

# scale the matrix to the properly scaled errors 'orbit_err'
scale = np.matmul((orbit_err / orbit_err_unscale).T, orbit_err / orbit_err_unscale)
orbit_covar_scale = scale * orbit_covar_unscale

# monte carlo switches. If monte_carlo, at least one of the others needs to be true.
# If not monte_carlo, the others are ignored
monte_carlo = True
N = 1000
perturb_orbit = True
perturb_spectra = True

# do you want a (static) third component to be found?
thirdlight = False

# lightfactors of your components (if thirdlight, give three)
lfs = [0.6173, 0.3827]

# sampling of your spectra in angstrom
sampling = 0.075

# enter wavelength range(s) in angstrom and give name to the line. Must be a dict.
lines = dict()
# lines['HeI4009'] = (4002, 4016)
# lines['HeI+II4026'] = (4018, 4033)
lines['Hdelta'] = (4086.7, 4111.3)
# lines['HeI4121'] = (4117, 4125)
# lines['HeI4143'] = (4135, 4149)
lines['HeII4200'] = (4192.3, 4207.0)
lines['Hgamma'] = (4324.2, 4352.5)
# lines['HeI4387'] = (4377, 4398)
# lines['HeI4471'] = (4465, 4477)
# lines['FeII4584'] = (4578, 4589)
lines['HeII4541'] = (4532.4, 4550.5)
lines['HeII4686'] = (4679.7, 4691.4)
# lines['HeI4713'] = (4707, 4720)
lines['Hbeta'] = (4841.6, 4878.0)
# lines['FeII5167'] = (5162, 5175)
# lines['FeII5198'] = (5190, 5205)
# lines['FeII5233'] = (5225, 5238)
# lines['FeII5276'] = (5270, 5282)
# lines['FeII5316+SII5320'] = (5310, 5325)
# lines['FeII5362'] = (5356, 5368)
lines['HeII5411'] = (5396.4, 5426.2)
# lines['OIII5592'] = (5584, 5600)
# lines['CIII5696'] = (5680, 5712)
# lines['FeII5780'] = (5770, 5790)
# lines['CIV5801+12'] = (5798, 5817)
lines['HeI5875'] = (5869.4, 5881.1)
# lines['Halpha'] = (6545, 6580)
# lines['HeI6678'] = (6674, 6682)
# lines['OI8446'] = (8437, 8455)

############################################################
# Here we start our experiment/runs


def run_join_threads(threads):
    """
    runs and joins the threads passes in list 'thread'. Also catched any exceptions raised within the threads.
    :param threads: list of threads to start and join when completed
    """
    for thread in threads:
        try:
            thread.start()
        except Exception as e:
            print(repr(thread), e)

    for thread in threads:
        thread.join()


# record starting time
starttime = time.time()
# print some info for reference
print('starting setup...')
print('orbit is:', orbit)
print('lightfactors are:', lfs)
print('spectroscopy folder is {}\n'.format(spec_folder))

# get all files in spectroscopy directory
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

# make working directories
pathlib.Path(fd3folder).mkdir(parents=True, exist_ok=True)

# save the run parameters for later reference
with open(gridfd3folder + "/params.txt", 'w') as paramfile:
    paramfile.write('orbit\t' + str(orbit) + '\n')
    paramfile.write('orbit_err\t' + str(orbit_err) + '\n')
    paramfile.write('lightfactors\t' + str(lfs) + '\n')
    paramfile.write('sampling\t' + str(sampling) + '\n')
    paramfile.write('k1s\t' + k1str + '\n')
    paramfile.write('k2s\t' + k2str + '\n')
    paramfile.write('spectra\t' + str(spectra_set) + '\n')
    paramfile.write('perturb orbit\t' + str(perturb_orbit) + '\n')
    paramfile.write('perturb spectra\t' + str(perturb_spectra) + '\n')
    paramfile.write('covar matrix\t' + str(orbit_covar_scale) + '\n')

# build gridfd3line objects
fd3lineobjects = list()
print('building fd3gridline object for:')
for line in lines.keys():
    print(' {}'.format(line))
    fd3lineobjects.append(
        fd3classes.Fd3class(line, lines[line], sampling, allfiles, thirdlight, orbit,
                            orbit_err, orbcovar=orbit_covar_scale, po=perturb_orbit,
                            ps=perturb_spectra, lfs=lfs, k1s=k1str, k2s=k2str))

# build threads around the lines
print('building threads')
cpus = 14
gridthreads = list()
d3threads = list()
for fd3line in fd3lineobjects:
    fd3line.set_k1(33)
    fd3line.set_k2(51)
    d3threads.append(fd3classes.Fd3Thread(fd3folder, fd3line))

# do an initial separation to renormalize on, we still consider this
run_join_threads(d3threads)
# recombine_and_renorm
print('renormalizing')
for fd3line in fd3lineobjects:
    fd3line.recombine_and_renorm()

if monte_carlo:
    # create threads
    print('number of threads will be {}'.format(cpus))
    print('each thread will have {} iterations to complete'.format(N / cpus))
    atleast = int(N / cpus)
    remainder = int(N % cpus)

    for i in range(remainder):
        gridthreads.append(fd3classes.GridFd3MCThread(gridfd3folder, i + 1, atleast + 1, fd3lineobjects))
    for i in range(remainder, cpus):
        gridthreads.append(fd3classes.GridFd3MCThread(gridfd3folder, i + 1, atleast, fd3lineobjects))

setuptime = time.time()
print('setup took {}s\n'.format(setuptime - starttime))
# start the MC gridfd3 process
print('starting runs!')
run_join_threads(gridthreads)
print('Thanks for your patience! You waited a whopping {} hours!'.format((time.time() - starttime) / 3600))
