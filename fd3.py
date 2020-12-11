"""
User script showing the typical use case for broad range fd3 separation.
First a list of files containing the relevant spectra is globbed.
Next, we specify the orbital parameters and the RV semi-amplitudes and various other
parameters such as the sampling rate and light factors
"""
import glob
import pathlib
import time
import matplotlib.pyplot as plt
import modules.gridfd3classes as fd3classes

# input
# define working directories
obj = '9_Sgr'
# gridfd3folder = obj + '/' + str(datetime.today().strftime('%y%m%dT%H%M%S'))
gridfd3folder = obj
fd3folder = gridfd3folder + '/fd3'

# find locations of your spectra
spectra_set = ['9_Sgr'
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
orbit = (3251, 56547, 0.648, 30.9, 31.0, 52.0)  # p, t0, e, omega(A), K1, K2

# lightfactors of your components (if thirdlight, give three)
lfs = [0.57, 0.43]
# do you want a (static) third component to be found?
thirdlight = False

# sampling of your spectra in angstrom
sampling = 0.03

# enter wavelength range(s) in natural log of wavelength and give name of line. Must be a dict.
lines = dict()
lines['range'] = (4010, 6800)
############################################################


def new_threads():
    d3threads.clear()
    for ffd3line in fd3lineobjects:
        d3threads.append(fd3classes.Fd3Thread(fd3folder, ffd3line))


def run_join_threads(threads):
    for thread in threads:
        try:
            thread.start()
        except Exception as e:
            print(repr(thread), e)

    for thread in threads:
        thread.join()


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
        fd3classes.Fd3class(line, lines[line], sampling, allfiles, thirdlight, orbit, lfs=lfs))

d3threads = list()
setuptime = time.time()
print('setup took {}s\n'.format(setuptime - starttime))
print('starting runs!')
now = time.time()
for line in fd3lineobjects:
    line.run_fd3(fd3folder)
for line in fd3lineobjects:
    # line.recombine_and_renorm()
    plt.title('k2 = {}'.format(orbit[5]))
    line.plot_fd3_results(offset=0.0)


print('Thanks for your patience! You waited a whopping {} hours!'.format((time.time() - starttime) / 3600))
plt.show()
