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

import numpy as np

import modules.spectra_manager as spec_man


class SpectrumError(Exception):
    """
    internal exception when something goes wrong loading spectra
    """

    def __init__(self, file, line, msg):
        super().__init__()
        print('Spectrum error for:', line, 'due to file', file, ':', msg)


class Fd3Exception(Exception):
    """
    internal exception when something goes wrong running gridf3 executable
    """

    def __init__(self, msg, line, threadno=None, iteration=None):
        super().__init__()
        if threadno is not None:
            print('Fd3Exception for:', line, 'in thread', threadno, 'iteration', iteration, ':', msg)
        else:
            print('Fd3Exception for:', line, ':', msg)


class Fd3gridLine:
    """
    defines one gridFd3 Job to be executed by a single binary call
    """

    def __init__(self, name, limits, samp, spectra_files, mc, tl, lfs, orb, orberr, po, ps, k1s, k2s, back=False):
        self.k1s = k1s
        self.k2s = k2s
        self.po = po
        self.ps = ps
        self.mc = mc
        self.tl = tl
        self.lfs = lfs
        self.orb = orb
        self.orberr = orberr
        self.name = name
        self.limits = np.log(limits)
        self.base = np.arange(self.limits[0] - 0.0001, self.limits[1] + 0.0001, samp)
        self.edgepoints = int(np.floor(0.0001 / samp))
        self.data = list()
        self.noises = list()
        self.mjds = list()
        self.spectra = spectra_files
        self.dof = 0
        self.used_spectra = 0
        self.back = back

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
        if self.used_spectra < 1:
            print(' {} has no spectral data, skipping'.format(repr(self)))
            return
        if not iteration:
            print(' making in file for {}'.format(repr(self)))
        self._make_infile(wd)
        if not iteration:
            print(' making master file for {}'.format(repr(self)))
        self._make_masterfile(wd)
        if not iteration:
            print(' running gridfd3 for {}'.format(repr(self)))
        self._run_fd3grid(wd)
        if not iteration:
            print(' saving output for {}'.format(repr(self)))
        self._save_output(wd, iteration)

    def _set_spectra(self):
        self.data = list()
        for j in range(len(self.spectra)):
            try:
                fluxhere, noisehere, mjdhere = spec_man.getspectrum(repr(self), self.spectra[j], self.base,
                                                                    self.edgepoints)
            except SpectrumError:
                continue
            self.data.append(fluxhere)
            self.noises.append(noisehere)
            self.mjds.append(mjdhere)
            self.used_spectra += 1
        self.data = np.array(self.data)
        self.data.setflags(write=False)  # make sure the original data is immutable!
        self.dof = self.used_spectra * len(self.base)
        print(' {} uses {} spectra'.format(self.name, self.used_spectra))

    def _perturb_spectra(self):
        newdata = np.copy(self.data)
        n = newdata.shape[1]
        m = newdata.shape[0]
        # perturb data
        pert = np.random.default_rng().normal(loc=0, scale=self.noises, size=(n, m))
        return newdata + pert.T

    def _perturb_orbit(self):
        return self.orb + np.random.default_rng().normal(0, self.orberr, 4)

    def _make_infile(self, wd):
        with open(wd + '/in{}'.format(self.name), 'w') as infile:
            # write first line
            infile.write(wd + "/{} ".format(self.name))
            infile.write("{} ".format(self.limits[0]))
            infile.write("{} ".format(self.limits[1]))
            # write whether you want the model spectra or not
            if self.back:
                infile.write("1 \n")
            else:
                infile.write("0 \n")
            # write the star switches
            if self.tl:
                infile.write('1 1 1 \n')
            else:
                infile.write('1 1 0 \n')
            # write observation data
            for j in range(self.used_spectra):
                if self.tl:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1],
                                                                      self.lfs[2]))
                # mjd, correction, noise, lfA, lfB, lfC
                else:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1]))
                # mjd, correction, noise, lfA, lfB
            if self.mc and self.po:
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

    def _make_masterfile(self, wd):
        with open(wd + '/{}.obs'.format(self.name), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(self.used_spectra + 1, len(self.base)))
            master = [self.base]
            if self.mc and self.ps:
                data = self._perturb_spectra()
            else:
                data = self.data
            for ii in range(len(data)):
                master.append(data[ii])
            towrite = np.array(master).T
            for ii in range(len(towrite)):
                obsfile.write(" ".join([str(num) for num in towrite[ii]]))
                obsfile.write('\n')

    def _run_fd3grid(self, wd):
        with open(wd + '/in{}'.format(self.name)) as inpipe, open(wd + '/out{}'.format(self.name), 'w') as outpipe:
            sp.run(['./bin/fd3grid'], stdin=inpipe, stdout=outpipe)

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

    def __repr__(self):
        return self.name


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
            raise Fd3Exception(e, ffd3line, self.threadno, ii)


class Fd3gridThread(threading.Thread):
    """
    defines a thread that runs its single fd3line object.
    """

    def __init__(self, fd3folder, fd3gridline: Fd3gridLine):
        super().__init__()
        self.fd3gridline = fd3gridline
        self.wd = fd3folder

    def run(self):
        """
        runs the fd3gridline disentangling
        """
        try:
            self.fd3gridline.run(self.wd)
        except FileNotFoundError as e:
            print('Exception occured when running gridfd3 for {}:'.format(repr(self.fd3gridline)), e)
            raise Fd3Exception(e, self.fd3gridline)
