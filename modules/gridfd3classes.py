"""
Defines the Fd3gridline object and its MCMC brother
"""
import os
import re
import shutil
import subprocess as sp
import threading
import time
import typing

import scipy.interpolate as spint
import scipy.optimize as spopt
import numpy as np
import matplotlib.pyplot as plt

import modules.spectra_manager as spec_man


def doppler_shift(w, rv):
    """
    doppler_shift in km/s
    """
    c = 299792.458
    return w * c / (c - rv)


class Fd3class:

    def __init__(self, name, limits, samp, spectra_files, tl, lfs, orb, orberr):
        self.tl = tl
        self.lfs = lfs
        self.orb = orb
        self.orberr = orberr
        self.name = name
        self.loglimits = np.log(limits)
        self.logbase = np.arange(self.loglimits[0] - 0.0001, self.loglimits[1] + 0.0001, samp)
        self.edgepoints = int(np.floor(0.0001 / samp))
        self.data = None
        self.noises = None
        self.mjds = None
        self.splines = list()
        self.spectra = spectra_files
        self.dof = 0
        self.no_used_spectra = 0

    def __repr__(self):
        return self.name

    def ecc_anom_of_phase(self, ph):
        # define keplers equation as function of a phase
        def keplers_eq(p):
            # build a function object that should be zero for a certain eccentric anomaly
            def kepler(ecc_an):
                return ecc_an - self.orb[2] * np.sin(ecc_an) - 2 * np.pi * p

            return kepler

        # find the root of keplers_eq(phase), which by construction returns a function for which the eccentric anomaly
        # is the independent variable.
        # current root finding algorithm is toms748, as it has the best convergence (2.7 bits per function evaluation)
        ph = np.remainder(ph, 1)
        return spopt.root_scalar(keplers_eq(ph), method='toms748',
                                 bracket=(0, 2 * np.pi)).root

    def true_anom(self, ph):
        E = self.ecc_anom_of_phase(ph)
        return 2 * np.arctan(np.sqrt((1 + self.orb[2]) / (1 - self.orb[2])) * np.tan(E / 2))

    def _set_spectra(self):
        self.data = list()
        self.noises = list()
        self.mjds = list()
        for j in range(len(self.spectra)):
            try:
                fluxhere, noisehere, mjdhere, spline = spec_man.getspectrum(repr(self), self.spectra[j], self.logbase,
                                                                            self.edgepoints)
            except spec_man.SpectrumError:
                continue
            self.splines.append(spline)
            self.data.append(fluxhere)
            self.noises.append(noisehere)
            self.mjds.append(mjdhere)
            self.no_used_spectra += 1
        self.data = np.array(self.data)
        self.mjds = np.array(self.mjds)
        self.noises = np.array(self.noises)
        self.dof = self.no_used_spectra * len(self.logbase)
        print(' {} uses {} spectra'.format(self.name, self.no_used_spectra))

    def _perturb_spectra(self):
        newdata = np.copy(self.data)
        n = newdata.shape[1]
        m = newdata.shape[0]
        pert = np.random.default_rng().normal(loc=0, scale=self.noises, size=(n, m))
        return newdata + pert.T

    def _perturb_orbit(self):
        return self.orb + np.random.default_rng().normal(0, self.orberr, 4)

    def run(self, wd):
        raise NotImplementedError

    def _make_masterfile(self, wd, data):
        with open(wd + '/{}.obs'.format(self.name), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(self.no_used_spectra + 1, len(self.logbase)))
            master = [self.logbase]
            for ii in range(len(data)):
                master.append(data[ii])
            towrite = np.array(master).T
            for ii in range(len(towrite)):
                obsfile.write(" ".join([str(num) for num in towrite[ii]]))
                obsfile.write('\n')


class Fd3Line(Fd3class):
    """
    Class that represents a line to be disentangled with fd3
    """

    def __init__(self, name, limits, samp, spectra_files, tl, lfs, orb, orberr, k1, k2):
        super().__init__(name, limits, samp, spectra_files, tl, lfs, orb, orberr)
        self.k1 = k1
        self.k2 = k2

    def run(self, wd):
        """
        do the grid minimization.
        1. write infile
        2. write obsfile for fd3
        3. run the executable
        :param wd: working directory
        """
        print(' fetching spectrum data for {}'.format(repr(self)))
        self._set_spectra()
        if self.no_used_spectra < 1:
            print(' {} has no spectral data, skipping'.format(repr(self)))
            return
        print(' making in file for {}'.format(repr(self)))
        self._make_infile(wd)
        print(' making master file for {}'.format(repr(self)))
        self._make_masterfile(wd, self.data)
        print(' running fd3 for {}'.format(repr(self)))
        self._run_fd3(wd)
        self._save_spectra(wd)

    def _save_spectra(self, wd):
        x = np.loadtxt(wd + '/products{}.mod'.format(self.name)).T
        x[0] = np.exp(x[0])
        error = np.average(self.noises)
        np.savetxt(wd + '/primary.txt', np.array([x[0], x[1], error * np.ones(len(x[1]))]).T)
        np.savetxt(wd + '/secondary.txt', np.array([x[0], x[2], error * np.ones(len(x[2]))]).T)
        x[1] *= self.lfs[0]
        x[2] *= self.lfs[1]
        inds = abs(x[1] + x[2] - 1) < 0.02
        x_avg = (x[1][inds] + 1 - x[2][inds]) / 2
        spline = spint.UnivariateSpline(x[0][inds], x_avg, s=0.05)
        x1 = x[1] - spline(x[0])
        x2 = x[2] - 1 + spline(x[0])
        np.savetxt(wd + '/primary_norm.txt',
                   np.array([x[0], x1 / self.lfs[0] + 1, error * np.ones(len(x[1]))]).T)
        np.savetxt(wd + '/secondary_norm.txt',
                   np.array([x[0], x2 / self.lfs[1] + 1, error * np.ones(len(x[2]))]).T)

    def _make_infile(self, wd):
        with open(wd + '/in{}'.format(self.name), 'w') as infile:
            # write first line
            infile.write(wd + "/{}.obs ".format(self.name))
            infile.write("{} ".format(self.loglimits[0]))
            infile.write("{} ".format(self.loglimits[1]))
            infile.write("{} ".format(wd + '/products{} '.format(self.name)))
            if self.tl:
                infile.write("1 1 1 \n\n")
            else:
                infile.write("1 1 0 \n\n")

            # write observation data
            for j in range(self.no_used_spectra):
                if self.tl:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {} {}\n'.format(
                            self.noises[j], self.lfs[0], self.lfs[1], self.lfs[2]))  # correction, noise, lfA, lfB, lfC
                else:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {} \n'.format(self.noises[j], self.lfs[0], self.lfs[1]))
            infile.write('\n')
            # write the AB-C orbital params
            infile.write('1 0     1 0    0 0    0 0    0 0    0 0 \n\n')
            # write the A-B orbital params
            infile.write(
                '{} 0 {} 0 {} 0 {} 0 {} 0 {} 0 0 0 \n\n'.format(self.orb[0], self.orb[1], self.orb[2], self.orb[3],
                                                                self.k1, self.k2))
            # write optimization params (won't be used tho)
            infile.write('100  1000  0.00001\n')

    def _run_fd3(self, wd):
        with open(wd + '/in{}'.format(self.name)) as inpipe, open(wd + '/out{}'.format(self.name), 'w') as outpipe:
            sp.run(['./bin/fd3'], stdin=inpipe, stdout=outpipe)

    def recombine(self, wd, eval_base, k1, k2):
        primary = np.loadtxt(wd + '/primary_norm.txt').T
        secondary = np.loadtxt(wd + '/secondary_norm.txt').T
        for i in range(self.no_used_spectra):
            print(i, '\r')
            phase = np.remainder((self.mjds[i] - self.orb[1]) / self.orb[0], 1)
            t = self.true_anom(phase)
            rvk1 = - k1 * (np.cos(t + np.pi / 180 * self.orb[3]) + self.orb[2] * np.cos(
                np.pi / 180 * self.orb[3]))
            rvk2 = k2 * (np.cos(t + np.pi / 180 * self.orb[3]) + self.orb[2] * np.cos(
                np.pi / 180 * self.orb[3]))

            shifted_primary_spline = spint.splrep(doppler_shift(primary[0], rvk1), primary[1])
            shifted_secondary_spline = spint.splrep(doppler_shift(secondary[0], rvk2), secondary[1])
            resamp_shifted_primary = spint.splev(eval_base, shifted_primary_spline)
            resamp_shifted_secondary = spint.splev(eval_base, shifted_secondary_spline)
            resamp_composite = spint.splev(eval_base, self.splines[i])
            reconstructees = self.lfs[0] * resamp_shifted_primary + self.lfs[1] * resamp_shifted_secondary
            residual = resamp_composite - reconstructees
            print('spectrum', i, 'has an average residual of', np.average(residual))


class Fd3gridLine(Fd3class):
    """
    defines one gridFd3 Job to be executed by a single binary call
    """

    def __init__(self, name, limits, samp, spectra_files, tl, lfs, orb, orberr, po, ps, k1s, k2s):
        super().__init__(name, limits, samp, spectra_files, tl, lfs, orb, orberr)
        self.po = po
        self.ps = ps
        self.k1s = k1s
        self.k2s = k2s

    def run(self, wd, iteration: int = None):
        """
        do the grid minimization.
        1. write infile
        2. write obsfile for fd3
        3. run the executable
        4. save output in handy-dandy npz files for later handling
        :param wd: working directory
        :param iteration: if an MCMC is running, which iteration are we doing
        """
        if not iteration or iteration == 1:
            print(' fetching spectrum data for {}'.format(repr(self)))
            self._set_spectra()
        if self.no_used_spectra < 1:
            print(' {} has no spectral data, skipping'.format(repr(self)))
            return
        if not iteration:
            print(' making in file for {}'.format(repr(self)))
        self._make_infile(wd)
        if not iteration:
            print(' making master file for {}'.format(repr(self)))
        if self.ps:
            data = self._perturb_spectra()
        else:
            data = self.data
        self._make_masterfile(wd, data)
        if not iteration:
            print(' running gridfd3 for {}'.format(repr(self)))
        self._run_fd3grid(wd)
        if not iteration:
            print(' saving output for {}'.format(repr(self)))
        self._save_output(wd, iteration)

    def _make_infile(self, wd):
        with open(wd + '/in{}'.format(self.name), 'w') as infile:
            # write first line
            infile.write(wd + "/{} ".format(self.name))
            infile.write("{} ".format(self.loglimits[0]))
            infile.write("{} ".format(self.loglimits[1]))
            # write the star switches
            if self.tl:
                infile.write('1 1 1 \n')
            else:
                infile.write('1 1 0 \n')
            # write observation data
            for j in range(self.no_used_spectra):
                if self.tl:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1],
                                                                      self.lfs[2]))
                # mjd, correction, noise, lfA, lfB, lfC
                else:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1]))
                # mjd, correction, noise, lfA, lfB
            if self.po:
                params = self._perturb_orbit()
            else:
                params = self.orb
            infile.write('1 0 0 0 0 0 \n')  # dummy parameters for the wide AB--C orbit
            # write the A-B orbital params
            infile.write('{} {} {} {} 0\n'.format(params[0], params[1], params[2], params[3]))  # 0 is the Deltaomega
            # write rv ranges and step size
            infile.write('{}\n'.format(self.k1s))
            infile.write('{}\n'.format(self.k2s))
            infile.write('{}\n'.format(self.dof))

    def _run_fd3grid(self, wd):
        with open(wd + '/in{}'.format(self.name)) as inpipe, open(wd + '/out{}'.format(self.name), 'w') as outpipe:
            sp.run(['./bin/gridfd3'], stdin=inpipe, stdout=outpipe)

    def _save_output(self, wd, iteration):
        with open(wd + '/out{}'.format(self.name)) as f:
            llines = f.readlines()
            llines.pop(0)
            kk1s = np.zeros(len(llines))
            kk2s = np.zeros(len(llines))
            cchisq = np.zeros(len(llines))
            for j in range(len(llines)):
                lline = re.split('[ ]|(?![\d.])', llines[j])
                kk1s[j] = np.float64(lline[0])
                kk2s[j] = np.float64(lline[1])
                cchisq[j] = np.float64(lline[2])
        chisqdir = wd + '/chisqs'
        if not os.path.isdir(chisqdir):
            os.mkdir(chisqdir)
        np.savez(chisqdir + '/chisq{}{}'.format(self.name, iteration if iteration is not None else ''), k1s=kk1s,
                 k2s=kk2s, chisq=cchisq)


class Fd3gridMCThread(threading.Thread):
    """
    defines an MCMC thread that runs its containing fd3gridlines for some specified number of iterations.
    """

    def __init__(self, fd3folder, threadno, iterations, fd3gridlines: typing.List[Fd3gridLine]):
        super().__init__()
        self.threadno = threadno
        self.wd = fd3folder + "/thread" + str(threadno)
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
        """
        Run this thread for its specified number of iterations.
        """
        ii = 0
        ffd3line = None
        try:
            for ii in range(self.iterations):
                # execute fd3gridline runs
                print('Thread {} running gridfd3 iteration {}...'.format(self.threadno, ii + 1))
                for ffd3line in self.fd3gridlines:
                    ffd3line.run(self.wd, ii + 1)
                print('estimated time to completion of thread {}: {}h'.format(self.threadno,
                                                                              (time.time() - self.threadtime) * (
                                                                                      self.iterations - ii - 1) / 3600))
                self.threadtime = time.time()
        except Exception as e:
            print('Exception occured when running gridfd3 for thread {}:'.format(self.threadno), e)


class Fd3ClassThread(threading.Thread):
    """
    defines a thread that runs its single fd3Class object.
    """

    def __init__(self, fd3folder, fd3obj: Fd3class):
        super().__init__()
        self.fd3obj = fd3obj
        self.wd = fd3folder

    def run(self):
        """
        runs the fd3gridline disentangling
        """
        try:
            self.fd3obj.run(self.wd)
        except FileNotFoundError as e:
            print('Exception occured when running gridfd3 for {}:'.format(repr(self.fd3obj)), e)
