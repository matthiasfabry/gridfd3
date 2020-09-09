import glob
import sys

import astropy.io.fits as fits

import lmfit as lm
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as spint
import scipy.optimize as spopt

# import spectralFunctions as sf


wd = None
try:
    wd = sys.argv[1]
except IndexError:
    exit()
obj = '9_Sgr'

orbit_params = np.genfromtxt(wd + '/MODEL/orbit.txt', dtype=None, filling_values=np.nan, encoding='utf-8')
orbit = {}
for param in orbit_params:
    orbit[param[0]] = param[1]
print('orbit reading complete!\n')

lf = [0.6173, 1-0.6173]


def ecc_anom_of_phase(ph):
    # define keplers equation as function of a phase
    def keplers_eq(p):
        # build a function object that should be zero for a certain eccentric anomaly
        def kepler(ecc_an):
            return ecc_an - orbit['e'] * np.sin(ecc_an) - 2 * np.pi * p

        return kepler

    # find the root of keplers_eq(phase), which by construction returns a function for which the eccentric anomaly
    # is the independent variable.
    # current root finding algorithm is toms748, as it has the best convergence (2.7 bits per function evaluation)
    ph = np.remainder(ph, 1)
    return spopt.root_scalar(keplers_eq(ph), method='toms748',
                             bracket=(0, 2 * np.pi)).root


def true_anom(ph):
    E = ecc_anom_of_phase(ph)
    return 2 * np.arctan(np.sqrt((1 + orbit['e']) / (1 - orbit['e'])) * np.tan(E / 2))

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

sampling = 0.05  # angstrom
spectra_files = glob.glob('/Users/matthiaf/data/spectra/' + obj + '**.fits')
primary = np.loadtxt(wd + '/MODEL/primary_norm.txt').T
secondary = np.loadtxt(wd + '/MODEL/secondary_norm.txt').T
primary_spline = spint.splrep(primary[0], primary[1])
secondary_spline = spint.splrep(secondary[0], secondary[1])

eval_base = np.array([])
resamp_primary = np.array([])
resamp_secondary = np.array([])
phases = np.zeros(len(spectra_files))
rvk1s = np.zeros(len(spectra_files))
rvk2s = np.zeros(len(spectra_files))
spectrum_splines = [0] * len(spectra_files)
resamp_composites = [np.array([])] * len(spectra_files)
print('fetching spectra')
for i in range(len(spectra_files)):
    with fits.open(spectra_files[i]) as hdul:
        phases[i] = np.remainder((hdul[0].header['MJD-obs'] - orbit['t0']) / orbit['p'], 1)
        spectrum_splines[i] = (hdul['NORM_SPLINE'].data[0])

    t = true_anom(phases[i])
    rvk1s[i] = - orbit['k1'] * (np.cos(t + np.pi / 180 * orbit['omega']) + orbit['e'] * np.cos(
        np.pi / 180 * orbit['omega']))
    rvk2s[i] = orbit['k2'] * (np.cos(t + np.pi / 180 * orbit['omega']) + orbit['e'] * np.cos(
        np.pi / 180 * orbit['omega']))

print('resampling composites')
for line, lrange in sorted(lines.items(), key=lambda x: x[1][0]):
    eval_base_here = np.arange(lrange[0], lrange[1], sampling)
    eval_base = np.concatenate((eval_base, eval_base_here))
    resamp_primary = np.concatenate((resamp_primary, spint.splev(eval_base_here, primary_spline)))
    resamp_secondary = np.concatenate((resamp_secondary, spint.splev(eval_base_here, secondary_spline)))
    for i in range(len(spectra_files)):
        evals = spint.splev(eval_base_here, spectrum_splines[i])
        resamp_composites[i] = np.concatenate((resamp_composites[i], spint.splev(eval_base_here, spectrum_splines[i])))

resamp_composites = np.array(resamp_composites)

resamp_shifted_primaries = np.zeros((len(spectra_files), len(eval_base)))
resamp_shifted_secondaries = np.zeros((len(spectra_files), len(eval_base)))
reconstructees = np.zeros((len(spectra_files), len(eval_base)))

lm_params = lm.Parameters()
lm_params.add('gamma1', value=0)
lm_params.add('gamma2', value=0)


def func2min(pars):
    parvals = pars.valuesdict()

    for iii in range(len(spectra_files)):
        shifted_primary_lamdas = sf.doppler_shift(primary[0], rvk1s[iii] + parvals['gamma1'])
        shifted_secondary_lambdas = sf.doppler_shift(secondary[0], rvk2s[iii] + parvals['gamma2'])

        shifted_primary_spline = spint.splrep(shifted_primary_lamdas, primary[1])
        shifted_secondary_spline = spint.splrep(shifted_secondary_lambdas, secondary[1])
        resamp_shifted_primaries[iii] = spint.splev(eval_base, shifted_primary_spline)
        resamp_shifted_secondaries[iii] = spint.splev(eval_base, shifted_secondary_spline)

        reconstructees[iii] = lf[0]*resamp_shifted_primaries[iii] + lf[1]*resamp_shifted_secondaries[iii]

    return (resamp_composites - reconstructees) / np.sqrt(reconstructees)


def do_min_mc():
    lm_minimizer = lm.Minimizer(func2min, lm_params)
    print('minimizing for gamma\'s')
    result = lm_minimizer.minimize()
    lm.report_fit(result.params)
    print(result.chisqr)
    # mcresult = lm.minimize(func2min, result.params, method='emcee', workers=os.cpu_count(), steps=1000)
    # lm.report_fit(mcresult.params)
    # corner.corner(mcresult.flatchain, labels=[r'$\gamma_1$', r'$\gamma_2$'],
    #               truths=list(mcresult.params.valuesdict().values()))


print('starting reconstruction')
do_min = False
if do_min:
    do_min_mc()
else:
    func2min(lm_params)
print('plotting')
for line in lines.keys():
    low = np.min(resamp_composites)
    print(low)
    high = np.max(resamp_composites)
    print(high)
    linedepth = high-low
    num = 4
    fig, axs = plt.subplots(nrows=2, ncols=num+1, sharex='all', sharey='row', gridspec_kw={'height_ratios': [3, 1]})
    for i in range(num+1):
        p = 1 / num * i
        ix = 0
        for ii in range(1, len(phases)):
            if abs(phases[ii] - p) < abs(phases[ix] - p):
                ix = ii
        axs[0][i].plot(eval_base, resamp_composites[ix], color='k', label='original')
        axs[0][i].plot(eval_base, resamp_shifted_primaries[i], color='b', label='primary')
        axs[0][i].plot(eval_base, resamp_shifted_secondaries[i], color='r', label='secondary')
        axs[0][i].plot(eval_base, reconstructees[ix], color='g', label='reconstruction')
        axs[0][i].set_title(r'{}'.format(np.round(phases[ix], 2)))
        axs[0][i].set_xlim((lines[line][0], lines[line][-1]))
        axs[0][i].set_ylim(low - 0.03, 1 + 0.02)

        axs[1][i].plot(eval_base, resamp_composites[ix] - reconstructees[ix], color='b', label='orig. - recon.')
        rms = np.std(resamp_composites[ix] - reconstructees[ix])
        mn = np.mean(resamp_composites[ix] - reconstructees[ix])
        print('residual mean, rms at phase:', mn, rms, phases[ix])
        axs[1][i].plot(eval_base, np.zeros(len(eval_base)), color='k')
        axs[1][i].set_ylim((-3*rms, 3*rms))

    axs[0][num].legend(loc=4)
    axs[1][num].legend(loc=4)
    axs[1][2].set_xlabel(r'$\lambda(\si{\angstrom})$')
    axs[0][0].set_ylabel(r'normalized flux')
    axs[1][0].set_ylabel(r'residuals')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.1)
    plt.savefig(wd + '/MODEL/recon{}.png'.format(line), dpi=200)
