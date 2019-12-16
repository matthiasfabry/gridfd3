import glob
import os
import re
import shutil
import subprocess as sp
import sys
import threading
import time
import typing

import astropy.io.fits as fits
import numpy as np
import scipy.interpolate as spint

starttime = time.time()


class Fd3gridLine:

    def __init__(self, name, limits):
        self.name = name
        self.files = list()
        self.limits = limits
        self.base = np.arange(limits[0] - 0.0001, limits[1] + 0.0001, 2e-5)
        self.data = list()
        self.noises = list()
        self.mjds = list()
        for j in range(number_of_files):
            with fits.open(allfiles[j]) as hdul:
                try:
                    spec_hdu = hdul['NORM_SPECTRUM']
                except KeyError:
                    print(allfiles[j], 'has no normalized spectrum, skipping')
                    continue
                loglamb = spec_hdu.data['log_wave']
                # check whether base is completely covered
                if loglamb[0] >= self.base[0] or loglamb[-1] <= self.base[-1]:
                    continue
                # check whether spline is present
                try:
                    hdul['NORM_SPLINE']
                except KeyError:
                    continue
                # determine indices where the line resides
                start = 0
                startline = 0
                endline = 0
                while loglamb[start] < self.base[0]:
                    start += 1
                    startline += 1
                    endline += 1
                # check if start of noise estimation interval is not zero
                if spec_hdu.data['norm_flux'][start] < 0.01:
                    continue
                while loglamb[startline] < limits[0]:
                    startline += 1
                    endline += 1
                # check if end of noise estimation interval is not zero or if no data in the estimation interval
                if spec_hdu.data['norm_flux'][startline] < 0.01 or start == startline:
                    continue
                while loglamb[endline] < limits[1]:
                    endline += 1
                # check whether end of line is not in interorder spacing of spectrograph
                if spec_hdu.data['norm_flux'][endline] < 0.01:
                    continue
                self.files.append(allfiles[j])
                # append in base evaluated flux values
                self.data.append(spint.splev(self.base, hdul['NORM_SPLINE'].data[0]))
                # determine noise near this line
                self.noises.append(np.std(hdul['NORM_SPECTRUM'].data['norm_flux'][start:startline]))
                # record mjd in separate list
                self.mjds.append(hdul[0].header['MJD-obs'])
        self.data = np.array(self.data)
        self.data.setflags(write=False)  # make sure the original data is immutable!
        print(' this line uses {} spectra'.format(len(self.files)))

    def run(self, wd, perturb: bool = False, iteration: int = None):
        self._make_infile(wd, perturb)
        self._make_masterfile(wd, perturb)
        self._run_fd3grid(wd)
        self._save_output(wd, iteration)

    def perturb_data(self):
        newdata = np.copy(self.data)
        n = len(newdata[:, 1])
        m = len(newdata[0]) - 1
        # perturb data
        pert = np.random.default_rng().normal(0, self.noises, (n, m))
        pert = np.hstack((np.zeros((n, 1)), pert))
        return newdata + pert

    @staticmethod
    def perturb_orbit():
        return orbit + np.random.default_rng().normal(0, orbit_err, 4)

    def _make_infile(self, wd, perturb: bool = False):
        with open(wd + '/in{}'.format(self.name), 'w') as infile:
            # write first line
            infile.write(wd + "/master{}.obs ".format(self.name))
            infile.write("{} ".format(self.limits[0]))
            infile.write("{} \n".format(self.limits[1]))
            # record mjds if we would need it later
            print('saving mjds')
            np.save('/Users/matthiasf/Software/fd3grid/' + wd + "/mjds{}".format(self.name), self.mjds)
            # write observation data
            for j in range(len(self.files)):
                infile.write(
                    str(self.mjds[j]) + ' 0 {} 1.0 1.0 1.0 \n'.format(
                        self.noises[j]))  # correction, noise, lfA, lfB, lfC
            infile.write('\n')
            # write the A-B orbital params
            if perturb:
                params = self.perturb_orbit()
            else:
                params = orbit
            infile.write('1 1 0 0 0 0 \n')  # dummy parameters for the wide AB--C orbit
            infile.write('{} {} {} {} 0\n\n'.format(params[0], params[1], params[2], params[3]))
            # write rv ranges and step size
            infile.write('11 55 1\n')
            infile.write('25 70 1\n')

    def _make_masterfile(self, wd, perturb: bool = False):
        with open(wd + '/master{}.obs'.format(self.name), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(len(self.files) + 1, len(self.base)))
            master = [self.base]
            if perturb:
                data = self.perturb_data()
            else:
                data = self.data
            for ii in range(len(data)):
                master.append(data[ii])
            towrite = np.array(master).T
            for ii in range(len(towrite)):
                obsfile.write(" ".join([str(num) for num in towrite[ii]]))
                obsfile.write('\n')

    def _run_fd3grid(self, wd):
        print('running fd3grid for {}'.format(self.name))
        with open(wd + '/in{}'.format(self.name)) as inpipe, open(wd + '/out{}'.format(self.name), 'w') as outpipe:
            sp.run(['./fd3grid'], stdin=inpipe, stdout=outpipe)

    def _save_output(self, wd, iteration):
        with open(wd + '/out{}'.format(self.name), 'r') as f:
            llines = f.readlines()
            llines.pop(0)
            k1s = np.zeros(len(llines))
            k2s = np.zeros(len(llines))
            chisq = np.zeros(len(llines))
            for j in range(len(llines)):
                lline = re.split('[ ]|(?![\d.])', llines[j])
                k1s[j] = np.float64(lline[0])
                k2s[j] = np.float64(lline[1])
                chisq[j] = np.float64(lline[2]) / (len(self.base) * len(self.files))
        chisqdir = wd + '/chisqs'
        if not os.path.isdir(chisqdir):
            os.mkdir(chisqdir)
        np.savez(chisqdir + '/chisq{}{}'.format(self.name, iteration if iteration is not None else ''), k1s=k1s,
                 k2s=k2s, chisq=chisq)


class Fd3gridThread(threading.Thread):

    def __init__(self, threadno, iterations, fd3gridlines: typing.List[Fd3gridLine]):
        super().__init__()
        self.threadno = threadno
        self.wd = fd3_folder + "/thread" + str(threadno)
        self.fd3gridlines = fd3gridlines
        self.iterations = iterations
        self.threadtime = time.time()
        self.chisqs = list()
        print('Thread {} will execute {} iterations.'.format(self.threadno, self.iterations))
        # create directory for this thread
        try:
            shutil.rmtree(self.wd)
        except OSError:
            pass
        try:
            os.mkdir(self.wd)
        except FileExistsError:
            pass
        # create k1file, k2flie to put them empty if they existed
        with open(self.wd + '/k1file', 'w'), open(self.wd + '/k2file', 'w'):
            pass

    def run(self) -> None:
        try:
            for ii in range(self.iterations):
                # execute fd3gridline runs
                print('Thread {} running fd3grid Iteration {}...'.format(self.threadno, ii + 1))
                for ffd3line in self.fd3gridlines:
                    ffd3line.run(self.wd, True, ii + 1)
                print('estimated time to completion of thread {}: {}h'.format(self.threadno,
                                                                              (time.time() - self.threadtime) * (
                                                                                      self.iterations - ii - 1) / 3600))
                self.threadtime = time.time()
        except Exception as e:
            print('Exception occured:', e)


print('starting setup...')
obj = None
try:
    obj = sys.argv[1]
except IndexError:
    print('please give an object!')
    exit()

if not os.path.isdir(obj):
    os.mkdir(obj)

fd3_folder = obj
print('fd3_folder is {}'.format(fd3_folder))
spec_folder = None
try:
    spec_folder = glob.glob('/Users/matthiasf/Data/Spectra/' + obj)[0]
except IndexError:
    print('no spectra folder of object found')
    exit()

print('spectroscopy folder is {}'.format(spec_folder))
monte_carlo = False
try:
    monte_carlo = sys.argv[2] == 'True'
except IndexError:
    pass

# all fits files in this directory
allfiles = glob.glob(spec_folder + '/**/*.fits', recursive=True)
number_of_files = len(allfiles)

if number_of_files == 0:
    print('no spectra found')
    exit()

# give geometric orbit
orbit = (3246, 1349, 0.648, 31)  # p, t0, e, Omega(A)
orbit_err = (0.15, 2.5, 0.001, 0.5)

# enter ln(lambda/A) range and name of line
lines = dict()
lines['Hzeta'] = (8.2630, 8.2685)
# lines['Hepsilon'] = (8.2845, 8.2888)
lines['HeI+II4026'] = (8.2990, 8.302)
lines['Hdelta'] = (8.3170, 8.3215)
lines['SiIV4116'] = (8.3215, 8.3238)
# lines['HeII4200'] = (8.3412, 8.3444)
# lines['Hgamma'] = (8.3730, 8.3785)
# lines['HeI4471'] = (8.4047, 8.4064)
# lines['HeII4541'] = (8.4195, 8.4226)
# lines['NV4604+4620'] = (8.4338, 8.4390)
# lines['HeII4686'] = (8.4510, 8.4534)
# lines['Hbeta'] = (8.4860, 8.4920)
# lines['HeII5411'] = (8.5940, 8.5986)
# lines['OIII5592'] = (8.6281, 8.6300)
# lines['CIII5696'] = (8.6466, 8.6482)
# lines['FeII5780'] = (8.6617, 8.6627)
# lines['CIV5801'] = (8.6652, 8.6667)
# lines['CIV5812'] = (8.6668, 8.6685)
# lines['HeI5875'] = (8.6777, 8.6794)

K = len(lines)
fd3lines = list()
print('building fd3gridline object for:')
for line in lines.keys():
    print(' {}'.format(line))
    fd3lines.append(Fd3gridLine(line, lines[line]))

if monte_carlo:
    N = 1000
    # create threads
    cpus = os.cpu_count()
    print('number of threads will be {}'.format(cpus))
    print('each thread will have {} iterations to complete'.format(N / cpus))
    atleast = int(N / cpus)
    remainder = int(N % cpus)
    threads = list()
    for i in range(remainder):
        threads.append(Fd3gridThread(i + 1, int(N / cpus) + 1, fd3lines))
    for i in range(remainder, cpus):
        threads.append(Fd3gridThread(i + 1, int(N / cpus), fd3lines))

    setuptime = time.time()
    print('setup took {}s'.format(setuptime - starttime))

    for thread in threads:
        thread.run()

    for thread in threads:
        thread.join()

    print('All runs done in {}h, see you later!'.format((time.time() - starttime) / 3600))
else:
    setuptime = time.time()
    print('setup took {}s'.format(setuptime - starttime))
    for fd3line in fd3lines:
        print('handling {} line'.format(fd3line.name))
        fd3line.run(fd3_folder)
