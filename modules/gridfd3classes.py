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
import scipy.linalg as spalg
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

    def __init__(self, name, linlimits, linsamp, spectra_files, tl, orb, orberr=None, orbcovar=None, po=False, ps=False,
                 lfs=(0.5, 0.5), k1s=None, k2s=None, k1=None, k2=None):
        self.tl = tl
        self.lfs = lfs
        self.orb = orb
        self.orberr = orberr
        self.orbcovar = orbcovar
        self.name = name
        logsamp = linsamp / 4500
        self.loglimits = np.log(linlimits)
        self.wideloglimits = self.loglimits + 200 * logsamp * np.array([-1, 1])
        self.linbase = np.arange(linlimits[0] - 20 * linsamp, linlimits[-1] + 20 * linsamp, linsamp)
        self.logbase = np.arange(self.loglimits[0] - 20 * logsamp, self.loglimits[-1] + 20 * logsamp, logsamp)
        self.widelogbase = np.arange(self.wideloglimits[0] - 20 * logsamp, self.wideloglimits[-1] + 20 * logsamp,
                                     logsamp)
        self.edgepoints = 20
        self.data = None
        self.widedata = None
        self.noises = None
        self.mjds = None
        self.spectra = spectra_files
        self.dof = 0
        self.no_used_spectra = 0
        self.po = po
        self.ps = ps
        self.k1s = k1s
        self.k2s = k2s
        self.prim = None
        self.sec = None
        self.k1 = k1
        self.k2 = k2

    def set_k1(self, k1):
        self.k1 = k1

    def set_k2(self, k2):
        self.k2 = k2

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
        return spopt.root_scalar(keplers_eq(ph), method='toms748', bracket=(0, 2 * np.pi)).root

    def true_anom(self, ph):
        E = self.ecc_anom_of_phase(ph)
        return 2 * np.arctan(np.sqrt((1 + self.orb[2]) / (1 - self.orb[2])) * np.tan(E / 2))

    def _set_spectra(self):
        print(' fetching spectrum data for {}'.format(repr(self)))
        self.data = list()
        self.widedata = list()
        self.noises = list()
        self.mjds = list()
        for j in range(len(self.spectra)):
            try:
                fluxhere, noisehere, mjdhere = spec_man.getspectrum(repr(self), self.spectra[j], self.logbase,
                                                                    self.edgepoints)
                wideflux, _, _ = spec_man.getspectrum(repr(self), self.spectra[j], self.widelogbase, self.edgepoints)
            except spec_man.SpectrumError:
                continue
            self.data.append(fluxhere)
            self.widedata.append(wideflux)
            self.noises.append(noisehere)
            self.mjds.append(mjdhere)
        self.no_used_spectra = len(self.data)
        self.widedata = np.array(self.widedata)
        self.data = np.array(self.data)
        self.mjds = np.array(self.mjds)
        self.noises = np.array(self.noises)
        self.dof = self.no_used_spectra * len(self.logbase)
        print(' {} uses {} spectra'.format(repr(self), self.no_used_spectra))

    def _perturb_spectra(self):
        newdata = np.copy(self.data)
        n = newdata.shape[1]
        m = newdata.shape[0]
        pert = np.random.default_rng().normal(scale=self.noises, size=(n, m))
        return newdata + pert.T

    def _perturb_orbit(self):
        if self.orbcovar is None:
            turb = np.random.default_rng().normal(scale=self.orberr)
        else:
            turb = np.random.default_rng().normal(size=(4, 1))
            c = spalg.cholesky(self.orbcovar)
            turb = np.dot(c, turb).T
        return self.orb + turb[0]

    def run_gridfd3(self, wd, iteration: int = None):
        """
        do the grid minimization.
        1. write infile
        2. write obsfile for fd3
        3. run_fd3 the executable
        4. save output in speedy npz files for later handling
        :param wd: working directory
        :param iteration: if an MCMC is running, which iteration are we doing
        """
        if self.data is None or self.widedata is None:
            self._set_spectra()
        if self.no_used_spectra < 1:
            print(' {} has no spectral data, skipping'.format(repr(self)))
            return
        if not iteration:
            print(' making in file for {}'.format(repr(self)))
        self._make_gridfd3_infile(wd)
        if not iteration:
            print(' making master file for {}'.format(repr(self)))
        self._make_grid_masterfile(wd)
        if not iteration:
            print(' running gridfd3 for {}'.format(repr(self)))
        self._run_gridfd3(wd)
        if not iteration:
            print(' saving output for {}'.format(repr(self)))
        self._handle_gridfd3_output(wd, iteration)

    def run_fd3(self, wd):
        """
        do the minimization.
        1. write infile
        2. write obsfile for fd3
        3. run_fd3 the executable
        4. handle the output to speedy npz files
        :param wd: working directory
        """
        if self.data is None or self.widedata is None:
            self._set_spectra()
        if self.no_used_spectra < 1:
            print(' {} has no spectral data, skipping'.format(repr(self)))
            return
        print(' making in file for {}'.format(repr(self)))
        self._make_fd3_infile(wd)
        print(' making master file for {}'.format(repr(self)))
        self._make_fd3_masterfile(wd)
        print(' running fd3 for {}'.format(repr(self)))
        self._run_fd3(wd)
        self._handle_fd3_output(wd)

    def _make_gridfd3_infile(self, wd):
        with open(wd + '/in{}'.format(repr(self)), 'w') as infile:
            # write first line
            infile.write(wd + "/{} ".format(repr(self)))
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
                    infile.write(str(self.mjds[j]) + ' 0 {} {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1],
                                                                             self.lfs[2]))
                # mjd, correction, noise, lfA, lfB, lfC
                else:
                    infile.write(str(self.mjds[j]) + ' 0 {} {} {}\n'.format(self.noises[j], self.lfs[0],
                                                                          self.lfs[
                                                                              1]))  # mjd, correction, noise, lfA, lfB
            infile.write('1 1 0 0 0 0 \n')  # dummy parameters for the wide AB--C orbit
            if self.po:
                params = self._perturb_orbit()
            else:
                params = self.orb
            # write the A-B orbital params
            infile.write(
                '{} {} {} {} 0\n'.format(params[0], params[1], params[2],
                                         params[3]))  # 0 is the for the precession of the omega
            # write rv ranges and step size
            infile.write('{}\n'.format(self.k1s))
            infile.write('{}\n'.format(self.k2s))
            infile.write('{}\n'.format(self.dof))

    def _make_fd3_infile(self, wd):
        with open(wd + '/in{}'.format(repr(self)), 'w') as infile:
            # write first line
            infile.write(wd + "/{}.obs ".format(repr(self)))
            infile.write("{} ".format(self.wideloglimits[0]))
            infile.write("{} ".format(self.wideloglimits[1]))
            infile.write("{} ".format(wd + '/products{} '.format(repr(self))))
            if self.tl:
                infile.write("1 1 1 \n")
            else:
                infile.write("1 1 0 \n")

            # write observation data
            for j in range(self.no_used_spectra):
                if self.tl:
                    infile.write(str(self.mjds[j]) + ' 0 {} {} {} {}\n'
                                 .format(self.noises[j], self.lfs[0],
                                         self.lfs[1], self.lfs[2]))  # correction, noise, lfA, lfB, lfC
                else:
                    infile.write(str(self.mjds[j]) + ' 0 {} {} {} \n'.format(self.noises[j], self.lfs[0], self.lfs[1]))
            # write the AB-C orbital params
            infile.write('1 0   1 0   0 0   0 0   0 0   0 0\n\n')
            # write the A-B orbital params
            infile.write(
                '{} 0 {} 0 {} 0 {} 0 {} 0 {} 0 0 0 \n\n'.format(self.orb[0], self.orb[1], self.orb[2], self.orb[3],
                                                                self.k1, self.k2))

    def _make_grid_masterfile(self, wd):
        with open(wd + '/{}.obs'.format(repr(self)), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(self.no_used_spectra + 1, len(self.logbase)))
            master = [self.logbase]
            if self.ps:
                data = self._perturb_spectra()
            else:
                data = self.data
            for ii in range(len(data)):
                master.append(data[ii])
            towrite = np.array(master).T
            for ii in range(len(towrite)):
                obsfile.write(" ".join([str(num) for num in towrite[ii]]))
                obsfile.write('\n')

    def _make_fd3_masterfile(self, wd):
        with open(wd + '/{}.obs'.format(repr(self)), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(self.no_used_spectra + 1, len(self.widelogbase)))
            master = [self.widelogbase]
            for ii in range(len(self.widedata)):
                master.append(self.widedata[ii])
            towrite = np.array(master).T
            for ii in range(len(towrite)):
                obsfile.write(" ".join([str(num) for num in towrite[ii]]))
                obsfile.write('\n')

    def _run_gridfd3(self, wd):
        with open(wd + '/in{}'.format(repr(self))) as inpipe, open(wd + '/out{}'.format(repr(self)), 'w') as outpipe:
            sp.run(['./bin/gridfd3'], stdin=inpipe, stdout=outpipe)

    def _run_fd3(self, wd):
        with open(wd + '/in{}'.format(repr(self))) as inpipe, open(wd + '/out{}'.format(repr(self)), 'w') as outpipe:
            sp.run(['./bin/fd3'], stdin=inpipe, stdout=outpipe)

    def _handle_gridfd3_output(self, wd, iteration):
        with open(wd + '/out{}'.format(repr(self))) as f:
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
        np.savez(chisqdir + '/chisq{}{}'.format(repr(self), iteration if iteration is not None else ''), k1s=kk1s,
                 k2s=kk2s, chisq=cchisq)

    def _handle_fd3_output(self, wd):
        x = np.loadtxt(wd + '/products{}.mod'.format(repr(self))).T
        x[0] = np.exp(x[0])
        x[1] *= self.lfs[0]
        x[2] *= self.lfs[1]
        # plt.plot(x[0], x[1])
        # plt.plot(x[0], x[2])
        # plt.show()
        # print(x[0], x[1])
        inds = abs(x[1] + x[2] - 1) < 1.05 * max(self.noises)
        # plt.plot(x[0][inds], np.zeros(len(x[0][inds])), 'ro')
        x_avg = (x[1][inds] + 1 - x[2][inds]) / 2
        spline = spint.UnivariateSpline(x[0][inds], x_avg, s=0.4)
        # plt.plot(x[0], spline(x[0]))
        # plt.show()
        x1 = x[1] - spline(x[0])
        x2 = x[2] - 1 + spline(x[0])
        self.prim = np.array([x[0], x1 / self.lfs[0] + 1]).T
        self.sec = np.array([x[0], x2 / self.lfs[1] + 1]).T
        # plt.plot(x[0], x1 / self.lfs[0] + 1)
        # plt.plot(x[0], x2 / self.lfs[1] + 1)
        # plt.show()
        errorp = np.std(self.prim[:20])
        errors = np.std(self.sec[:20])
        np.savetxt(wd + '/{}primary.txt'.format(repr(self)), np.array([x[0], x[1], errorp * np.ones(len(x[1]))]).T)
        np.savetxt(wd + '/{}secondary.txt'.format(repr(self)), np.array([x[0], x[2], errors * np.ones(len(x[2]))]).T)
        np.savetxt(wd + '/{}primary_norm.txt'.format(repr(self)),
                   np.array([x[0], x1 / self.lfs[0] + 1, errorp * np.ones(len(x[1]))]).T)
        np.savetxt(wd + '/{}secondary_norm.txt'.format(repr(self)),
                   np.array([x[0], x2 / self.lfs[1] + 1, errors * np.ones(len(x[2]))]).T)

    def recombine_and_renorm(self):
        avres = np.zeros(self.no_used_spectra)
        for i in range(self.no_used_spectra):
            res = self._get_residual_and_norm(i)
            avres[i] = np.average(res)
        print('the std of the average residuals of all {} spectra is {}'.format(self.no_used_spectra, np.std(avres)))

    def _get_residual_and_norm(self, i):
        phase = np.remainder((self.mjds[i] - self.orb[1]) / self.orb[0], 1)
        t = self.true_anom(phase)
        rvk1 = - self.k1 * (np.cos(t + np.pi / 180 * self.orb[3]) + self.orb[2] * np.cos(np.pi / 180 * self.orb[3]))
        rvk2 = self.k2 * (np.cos(t + np.pi / 180 * self.orb[3]) + self.orb[2] * np.cos(np.pi / 180 * self.orb[3]))
        shifted_primary_spline = spint.splrep(doppler_shift(self.prim[:, 0], rvk1), self.prim[:, 1])
        shifted_secondary_spline = spint.splrep(doppler_shift(self.sec[:, 0], rvk2), self.sec[:, 1])
        resamp_shifted_primary = spint.splev(self.linbase, shifted_primary_spline)
        resamp_shifted_secondary = spint.splev(self.linbase, shifted_secondary_spline)
        resamp_composite = spint.splev(self.linbase, spint.splrep(np.exp(self.logbase), self.data[i, :]))
        reconstructees = self.lfs[0] * resamp_shifted_primary + self.lfs[1] * resamp_shifted_secondary
        residual = resamp_composite - reconstructees
        ll = len(residual)

        def corr(x):
            yleft = np.average(residual[:int(np.floor(0.1 * ll))])
            yright = np.average(residual[int(np.ceil(0.9 * ll)):])
            xleft = self.linbase[int(np.floor(0.05 * ll))]
            xright = self.linbase[int(np.ceil(0.95 * ll))]
            return yleft + (yright - yleft) / (xright - xleft) * (x - xleft)

        self.data[i, :] -= corr(np.exp(self.logbase))
        self.widedata[i, :] -= corr(np.exp(self.widelogbase))

        return residual

    def plot_fd3_results(self, ax=plt.gca(), offset=0):
        ax.plot(self.sec[:, 0], self.sec[:, 1], 'r', label='secondary')
        ax.plot(self.prim[:, 0], self.prim[:, 1] + offset, 'b', label='primary + {}'.format(offset))
        ax.grid()
        ax.legend()


class GridFd3MCThread(threading.Thread):
    """
    defines an MCMC thread that runs its containing fd3gridlines for some specified number of iterations.
    """

    def __init__(self, fd3folder, threadno, iterations, fd3gridlines: typing.List[Fd3class]):
        super().__init__()
        self.threadno = threadno
        self.wd = fd3folder + "/thread" + str(threadno)
        self.fd3gridlines = fd3gridlines
        self.iterations = iterations
        self.threadtime = 0
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
        # with open(self.wd + '/k1file', 'w'), open(self.wd + '/k2file', 'w'):
        #     pass

    def __repr__(self):
        return "GridFd3MC thread no " + self.threadno

    def run(self) -> None:
        """
        Run this thread for its specified number of iterations.
        """
        for ii in range(self.iterations):
            self.threadtime = time.time()
            # execute fd3gridline runs
            print('Thread {} running gridfd3 iteration {}...'.format(self.threadno, ii + 1))
            for ffd3line in self.fd3gridlines:
                ffd3line.run_gridfd3(self.wd, ii + 1)
            print('estimated time to completion of thread {}: {}h'.format(self.threadno,
                                                                          (time.time() - self.threadtime) * (
                                                                                  self.iterations - ii - 1) / 3600))


class GridFd3Thread(threading.Thread):
    """
    defines a thread that runs gridfd3 on its single fd3Class object.
    """

    def __init__(self, fd3folder, fd3obj: Fd3class):
        super().__init__()
        self.fd3obj = fd3obj
        self.wd = fd3folder

    def __repr__(self):
        return "GridFd3 thread for " + self.fd3obj.__repr__()

    def run(self):
        """
        runs the fd3gridline disentangling
        """
        self.fd3obj.run_gridfd3(self.wd)


class Fd3Thread(threading.Thread):
    """
    defines a thread that runs fd3 on its single fd3Class object.
    """

    def __init__(self, fd3folder, fd3obj: Fd3class):
        super().__init__()
        self.fd3obj = fd3obj
        self.wd = fd3folder

    def __repr__(self):
        return "Fd3Thread for " + self.fd3obj.__repr__()

    def run(self):
        """
        runs the fd3 line disentangling
        """
        self.fd3obj.run_fd3(self.wd)
