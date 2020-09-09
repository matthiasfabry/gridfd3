import astropy.io.fits as fits
import numpy as np
import scipy.interpolate as spint


class SpectrumError(Exception):
    """
    internal exception when something goes wrong loading spectra
    """

    def __init__(self, file, line, msg):
        super().__init__()
        print(' Spectrum error for:', line, 'due to file', file + ':', msg)


def getspectrum(line, file, lambdabase, edgepoints):
    """
    This function must return the fluxvalues evaluated in wavelength range lambdabase, a noise estimation and an mjd
    of the spectrum denoted by the parameter file. IF you determine this file does not contain a spectrum on lambdabase,
    raise a SpectrumException
    :param line: string representing the line you are trying to disentangle
    :param file: string that points to the file containing the spectrum (which you glob in the main gridfd3.py script)
    :param lambdabase: wavelength base (in log space) you must return the fluxvalues of
    :param edgepoints: number of points before the line that is used to estimate the noise
    :return: flux, noise, mjd as stated in the description of this function or None
    """
    with fits.open(file) as hdul:
        try:
            spec_hdu = hdul['NORM_SPECTRUM']
        except KeyError:
            raise SpectrumError(file, line, 'has no normalized spectrum, skipping')
        loglamb = spec_hdu.data['log_wave']
        # check whether base is completely covered
        if loglamb[0] > lambdabase[0] or loglamb[-1] < lambdabase[-1]:
            raise SpectrumError(file, line, 'does not cover line')
        try:
            spline = hdul['NORM_SPLINE']
            logspline = hdul['LOG_NORM_SPLINE']
        except KeyError:
            raise SpectrumError(file, line, 'has no spline')
        # append in base evaluated flux values
        flux = spint.splev(lambdabase, logspline.data[0])
        # determine noise near this line
        noise = np.std(flux[:edgepoints-1])
        # get mjd
        mjd = hdul[0].header['MJD-obs']
        return flux, noise, mjd, spline.data[0]


def getspectrumspline(line, file):
    with fits.open(file) as hdul:
        try:
            spl = hdul['NORM_SPLINE']
        except KeyError:
            raise SpectrumError(file, line, 'has no spline')
        mjd = hdul[0].header['MJD-obs']
        return spl.data[0], mjd
