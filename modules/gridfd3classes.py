import os
import re
import shutil
import subprocess as sp
import threading
import time
import typing

import astropy.io.fits as fits
import numpy as np
import scipy.interpolate as spint


class Fd3gridLine:

    def __init__(self, name, limits, samp, spectra_files, mc, tl, lfs, orb, orberr, po, ps, k1s, k2s):
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
        self.used_spectra = list()
        self.limits = limits
        self.base = np.arange(limits[0] - 0.0001, limits[1] + 0.0001, samp)
        self.data = list()
        self.noises = list()
        self.mjds = list()
        self.spectra = spectra_files
        self.dof = 0

    def set_spectra(self):
        self.data = list()
        for j in range(len(self.spectra)):
            with fits.open(self.spectra[j]) as hdul:
                try:
                    spec_hdu = hdul['NORM_SPECTRUM']
                except KeyError:
                    print(self.spectra[j], 'has no normalized spectrum, skipping')
                    continue
                loglamb = spec_hdu.data['log_wave']
                # check whether base is completely covered
                if loglamb[0] >= self.base[0] or loglamb[-1] <= self.base[-1]:
                    print(self.spectra[j], 'does not fully cover', self.name)
                    continue
                # check whether spline is present
                try:
                    hdul['LOG_NORM_SPLINE']
                except KeyError:
                    print(self.spectra[j], 'has no log_norm_spline')
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
                    print(self.spectra[j], 'has low start of noise interval in line', self.name)
                    continue
                while loglamb[startline] < self.limits[0]:
                    startline += 1
                    endline += 1
                # check if end of noise estimation interval is not zero or if no data in the estimation interval
                if spec_hdu.data['norm_flux'][startline] < 0.01 or start == startline:
                    print(self.spectra[j], 'has low end of noise interval in line', self.name)
                    continue
                while loglamb[endline] < self.limits[1]:
                    endline += 1
                # check whether end of line is not in interorder spacing of spectrograph
                if spec_hdu.data['norm_flux'][endline] < 0.01:
                    print(self.spectra[j], 'has low end of data interval in line', self.name)
                    continue
                self.used_spectra.append(self.spectra[j])
                # append in base evaluated flux values
                evals = spint.splev(self.base, hdul['log_NORM_SPLINE'].data[0])
                self.data.append(evals)
                # determine noise near this line
                self.noises.append(np.std(hdul['NORM_SPECTRUM'].data['norm_flux'][start:startline]))
                # record mjd in separate list
                self.mjds.append(hdul[0].header['MJD-obs'])
        self.data = np.array(self.data)
        self.data.setflags(write=False)  # make sure the original data is immutable!
        self.dof = len(self.used_spectra) * len(self.base)
        print(' this line uses {} spectra'.format(len(self.used_spectra)))

    def run(self, wd, iteration: int = None):
        if not iteration or iteration == 1:
            self.set_spectra()
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

    def perturb_spectra(self):
        newdata = np.copy(self.data)
        n = newdata.shape[1]
        m = newdata.shape[0]
        # perturb data
        pert = np.random.default_rng().normal(loc=0, scale=self.noises, size=(n, m))
        return newdata + pert.T

    def perturb_orbit(self):
        return self.orb + np.random.default_rng().normal(0, self.orberr, 4)

    def _make_infile(self, wd):
        with open(wd + '/in{}'.format(self.name), 'w') as infile:
            # write first line
            infile.write(wd + "/master{}.obs ".format(self.name))
            infile.write("{} ".format(self.limits[0]))
            infile.write("{} \n".format(self.limits[1]))
            # write the star switches
            if self.tl:
                infile.write('1 1 1 \n')
            else:
                infile.write('1 1 0 \n')
            # write observation data
            for j in range(len(self.used_spectra)):
                if self.tl:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1],
                                                                      self.lfs[2]))
                # correction, noise, lfA, lfB, lfC
                else:
                    infile.write(
                        str(self.mjds[j]) + ' 0 {} {} {}\n'.format(self.noises[j], self.lfs[0], self.lfs[1]))
                # correction, noise, lfA, lfB
            if self.mc and self.po:
                params = self.perturb_orbit()
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
        with open(wd + '/master{}.obs'.format(self.name), 'w') as obsfile:
            obsfile.write('# {} X {} \n'.format(len(self.used_spectra) + 1, len(self.base)))
            master = [self.base]
            if self.mc and self.ps:
                data = self.perturb_spectra()
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
            sp.run(['./fd3grid'], stdin=inpipe, stdout=outpipe)

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
        return self.name + 'line'


class Fd3gridMCThread(threading.Thread):

    def __init__(self, fd3folder, threadno, iterations, fd2gridlines: typing.List[Fd3gridLine]):
        super().__init__()
        self.threadno = threadno
        self.wd = fd3folder + "/thread" + str(threadno)
        self.fd2gridlines = fd2gridlines
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
                print('Thread {} running gridfd3 iteration {}...'.format(self.threadno, ii + 1))
                for ffd2line in self.fd2gridlines:
                    ffd2line.run(self.wd, ii + 1)
                print('estimated time to completion of thread {}: {}h'.format(self.threadno,
                                                                              (time.time() - self.threadtime) * (
                                                                                      self.iterations - ii - 1) / 3600))
                self.threadtime = time.time()
        except Exception as e:
            print('Exception occured when running gridfd3:', e)


class Fd3gridThread(threading.Thread):

    def __init__(self, fd3folder, fd3gridline: Fd3gridLine):
        super().__init__()
        self.fd3gridline = fd3gridline
        self.wd = fd3folder

    def run(self):
        try:
            self.fd3gridline.run(self.wd)
        except Exception as e:
            print('Exeption occured when running gridfd3 for {}:'.format(repr(self.fd3gridline)), e)
