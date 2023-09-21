import os
from glob import glob
import pathlib

import numpy as np
from numpy.polynomial import Polynomial

from astropy import units as u
from astropy import constants as const
from astropy.io import ascii

from synphot import SourceSpectrum, ReddeningLaw
from synphot.models import BlackBodyNorm1D
from synphot.units import convert_flux

import stsynphot as stsyn

# from pydwd.readspec import read_eso
# from pydwd.procspec import wavelength2vel

from matplotlib import pyplot as plt


# def read_full_spectrum(path, date, nsigma=3):
#     files = glob(os.path.join(path, "*"+date+"*.fits"))
#     wavelength = np.array([])
#     flux = np.array([])
#     dflux = np.array([])
#     for file in files:
#         curr_wavelength, curr_flux, curr_dflux, obj, obs_date, jd, bjd, barycentric_correction, instrument, filename = \
#                     read_eso(file, interesting_wavelengths=None, instruments=["UVES"])

#         idx = (curr_dflux < nsigma*np.nanmedian(curr_dflux)) & (curr_flux > 0) & ~np.isnan(curr_flux)
#         wavelength = np.concatenate([wavelength, curr_wavelength[idx]])
#         flux = np.concatenate([flux, curr_flux[idx]])
#         dflux = np.concatenate([dflux, curr_dflux[idx]])
    
#     # sort by wavelength
#     sort_idx = np.argsort(wavelength)
#     wavelength = wavelength[sort_idx]
#     flux = flux[sort_idx]
#     dflux = dflux[sort_idx]
    
#     return wavelength, flux, dflux


def bin_spectrum(wavelength, flux, binning=25, func=np.nanmedian, return_uncertainty=False):
    if hasattr(wavelength, 'unit'):
        wavelength = wavelength.value
    if hasattr(flux, 'unit'):
        flux = flux.value
    gap_idx = np.concatenate([[-1], np.where(np.diff(wavelength) > 10)[0], [len(wavelength)]])+1

    wavelength_binned = np.array([])
    flux_binned = np.array([])
    if return_uncertainty:
        flux_error = np.array([])
    for i in range(len(gap_idx)-1):
        curr_wavelength = wavelength[gap_idx[i]:gap_idx[i+1]]
        curr_flux = flux[gap_idx[i]:gap_idx[i+1]]

        nbins = int(np.round(len(curr_wavelength)/binning))
        bins = np.linspace(np.floor(curr_wavelength[0]), np.ceil(curr_wavelength[-1]), nbins)
        digitized = np.digitize(curr_wavelength, bins)
        bin_medians = np.array([func(curr_flux[digitized == x]) for x in range(len(bins))])
        wavelength_binned = np.concatenate([wavelength_binned, bins])
        flux_binned = np.concatenate([flux_binned, bin_medians])
#         idx = ~np.isnan(bin_medians)
#         wavelength_binned = np.concatenate([wavelength_binned, bins[idx]])
#         flux_binned = np.concatenate([flux_binned, bin_medians[idx]])
        
        if return_uncertainty:
            if func == np.nanmedian:
                N = np.array([np.count_nonzero(~np.isnan(curr_flux[digitized == x])) for x in range(len(bins))])
                bin_std = np.array([np.nanstd(curr_flux[digitized == x]) for x in range(len(bins))])
                bin_med_error = bin_std*np.sqrt(np.pi*N/2/(N-1))
                flux_error = np.concatenate([flux_error, bin_med_error])
    if return_uncertainty:
        return wavelength_binned, flux_binned, flux_error
    else:
        return wavelength_binned, flux_binned


def plot_full_spectrum(wavelength, flux, ax=None, **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
        
    gap_idx = np.concatenate([[-1], np.where(np.diff(wavelength) > 10)[0], [len(wavelength)]])+1

    for i in range(len(gap_idx)-1):
        curr_wavelength = wavelength[gap_idx[i]:gap_idx[i+1]]
        curr_flux = flux[gap_idx[i]:gap_idx[i+1]]
        
        ax.plot(curr_wavelength, curr_flux, **kwargs)
    
    return ax


def normalize_full_spectrum(wavelength, flux, npoly=1, semilog=True):
    gap_idx = np.concatenate([[-1], np.where(np.diff(wavelength) > 10)[0], [len(wavelength)]])+1

    flux_norm = np.array([])
    for i in range(len(gap_idx)-1):
        curr_wavelength = wavelength[gap_idx[i]:gap_idx[i+1]]
        curr_flux = flux[gap_idx[i]:gap_idx[i+1]]
        
        if semilog:
            curr_flux = np.log10(curr_flux)
            
        p = Polynomial.fit(curr_wavelength, curr_flux, npoly)
        flux_norm = np.concatenate([flux_norm, curr_flux/p(curr_wavelength)])
        
    if semilog:
        flux_norm = 10**flux_norm
    
    return flux_norm


# def read_files(files):
#     obs_date = [None]*len(files)
#     jd = [None]*len(files)
#     bjd = [None]*len(files)
#     barycentric_correction = [None]*len(files)

#     for i in range(len(files)):
#         wavelength, flux, dflux, obj, obs_date[i], jd[i], bjd[i], barycentric_correction[i], instrument, filename = \
#             read_eso(files[i], interesting_wavelengths=None, instruments=["UVES"])

#         if i == 0:
#             x = wavelength.value
#             y = np.zeros((len(files), len(x)))
#             y[i, :] = flux.value
#         else:
#             y[i, :] = np.interp(x, wavelength.value, flux.value)
            
#     return x, y, obs_date, bjd, barycentric_correction


# def read_files_in_restframe(files, K, phi, phi0, gamma):
#     obs_date = [None]*len(files)
#     jd = [None]*len(files)
#     bjd = [None]*len(files)
#     barycentric_correction = [None]*len(files)

#     for i in range(len(files)):
#         wavelength, flux, dflux, obj, obs_date[i], jd[i], bjd[i], barycentric_correction[i], instrument, filename = \
#             read_eso(files[i], interesting_wavelengths=None, instruments=["UVES"])
        
#         v = K * np.sin(2*np.pi*(phi[i] - phi0)) + gamma + barycentric_correction[i]
#         wavelength_rest = (wavelength/(1 + v/const.c)).to(u.angstrom)
        
#         if i == 0:
#             x = wavelength_rest.value
#             y = np.zeros((len(files), len(x)))
#             y[i, :] = flux.value
#         else:
#             y[i, :] = np.interp(x, wavelength_rest.value, flux.value)
            
#     return x, y, obs_date, bjd, barycentric_correction


# def read_files_in_restframe_using_rvs(files, v):
#     obs_date = [None]*len(files)
#     jd = [None]*len(files)
#     bjd = [None]*len(files)
#     barycentric_correction = [None]*len(files)

#     for i in range(len(files)):
#         wavelength, flux, dflux, obj, obs_date[i], jd[i], bjd[i], barycentric_correction[i], instrument, filename = \
#             read_eso(files[i], interesting_wavelengths=None, instruments=["UVES"])
        
#         v[i] = v[i] + barycentric_correction[i]
#         wavelength_rest = (wavelength/(1 + v[i]/const.c)).to(u.angstrom)
        
#         if i == 0:
#             x = wavelength_rest.value
#             y = np.zeros((len(files), len(x)))
#             y[i, :] = flux.value
#         else:
#             y[i, :] = np.interp(x, wavelength_rest.value, flux.value)
            
#     return x, y, obs_date, bjd, barycentric_correction


# def lambda2v(x, y, lambda0, barycentric_correction):
#     yv = y.copy()
#     for i in range(y.shape[0]):
#         curr_v, _ = wavelength2vel(x*u.Angstrom, lambda0=lambda0, barycentric_correction=barycentric_correction[i])
#         if i == 0:
#             v = curr_v
#         else:
#             yv[i, :] = np.interp(v.value, curr_v.value, y[i, :])
    
#     return v, yv


def normalize(x, y, xmin=-np.inf, xmax=np.inf, npoly=1):
    idx = (x >= xmin) & (x <= xmax)
    
    if y.ndim == 1:
        coef = np.polyfit(x[idx], y[idx], npoly)
        p = np.poly1d(coef)
        y_norm = y[idx]/p(x[idx])
    else:
        y_norm = np.zeros((y.shape[0], np.count_nonzero(idx)))
        for i in range(y.shape[0]):
            coef = np.polyfit(x[idx], y[i, idx], npoly)
            p = np.poly1d(coef)
            y_norm[i, :] = y[i, idx]/p(x[idx])
    return y_norm, x[idx]


def plotspec(y, sorted_idx=None, xmin=0, xmax=None, ymin=0, ymax=None,
             vmin=0.7, vmax=1.2, label=None, print_row_idx=False, figsize=(20,10)):
    if xmax is None:
        xmax = y.shape[1]
    if ymax is None:
        ymax = y.shape[0]
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.matshow(y[sorted_idx, :], fignum=0, vmin=vmin, vmax=vmax, aspect='auto', origin='lower', extent=[xmin, xmax, ymin, ymax])
    ax.xaxis.tick_bottom()
    ax.set_yticks(np.arange(0.5, y.shape[0]+0.5))
    if label is not None:
        if print_row_idx:
            ax.set_yticklabels([f"{label[i]}: {i}" for i in sorted_idx])
        else:
            ax.set_yticklabels([label[i] for i in sorted_idx])
    
    return fig, ax


def plotspec_phased(phase, y, xmin=0, xmax=None,
                    vmin=0.7, vmax=1.2, label=None, print_row_idx=False, figsize=(17,10), ax=None, bin_size=0.025):
    if xmax is None:
        xmax = y.shape[1]
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    sorted_idx = np.argsort(phase)
    
    phase_vec = np.arange(0, 1, bin_size)
    #phase_bin_idx = np.arange(0, len(phase_vec)+1, 5)
    y_even_spaced = np.zeros((len(phase_vec), y.shape[1]))*np.nan
    
    for i in range(len(phase)):
        #idx = np.where(np.abs(phase[i] - phase_vec) <= bin_size)[0]
        idx = np.argmin(np.abs(phase[i] - phase_vec))
        y_even_spaced[idx, :] = y[i, :]
    
    """
    # duplicate spectra to fill each 0.05 phase bin
    phase_bin_idx = np.arange(0, len(phase_vec)+1, 5)
    for i in range(len(phase_bin_idx) - 1):
        idx = np.arange(phase_bin_idx[i], phase_bin_idx[i+1])
        filled_lines = np.any(y_even_spaced[idx, :], axis=1)
        if filled_lines:
            empty_lines = np.where(~filled_lines)[0]
            filled_lines = np.where(filled_lines)[0]
            for empty_line in empty_lines:
                filled_line_idx = np.argmin(np.abs(empty_line - filled_lines))
                y_even_spaced[idx, :] = y_even_spaced[
     """
    
    plt.matshow(y_even_spaced, fignum=0, vmin=vmin, vmax=vmax, aspect='auto', origin='lower', extent=[xmin, xmax, 0, 1])
    ax.xaxis.tick_bottom()
    ax.set_yticks(np.arange(0, 1, 0.05))
    if label is not None:
        if print_row_idx:
            ax.set_yticklabels([f"{label[i]}: {i}" for i in sorted_idx])
        else:
            ax.set_yticklabels([label[i] for i in sorted_idx])
    
    if ax is None:
        return fig, ax
    
    
def blackbody(temperature, wavelength, ebv=None, extinction_model='mwavg'):
    bb = SourceSpectrum(BlackBodyNorm1D, temperature=temperature)  # [photons s^-1 cm^-2 A^-1]
    if ebv is not None:
        # apply extinction
        ext = ReddeningLaw.from_extinction_model(extinction_model).extinction_curve(ebv)
        bb = bb * ext
    bb = bb(wavelength)/(const.R_sun / const.kpc) ** 2  # undo synphot normalization (but leave the pi factor from integration over half a sphere)
    bb = convert_flux(wavelength, bb, 'flam')  # [flam] = [erg s^-1 cm^-2 A^-1]
    bb = bb.to(u.erg/u.s/u.cm**2/u.angstrom)  # express in normal astropy units
    return bb


def calc_pivot_wavelength(wavelength, bandpass):
    dlambda = np.diff(wavelength)
    dlambda = np.concatenate([dlambda, np.array([dlambda[-1]])])
    
    wl = np.sqrt(np.sum(dlambda*wavelength*bandpass)/np.sum(dlambda*wavelength**(-1)*bandpass))  # identical to SVO's pivot wavelength
    
    return wl


def calc_effective_wavelength(wavelength, bandpass):
    vega = ascii.read('../../../../../data/Filters/vega.dat', names=["wavelength", "flux"])
    vega.sort('wavelength')
    vega = np.interp(wavelength, vega['wavelength'], vega['flux'])
    
    dlambda = np.diff(wavelength)
    dlambda = np.concatenate([dlambda, np.array([dlambda[-1]])])
    
    wl = np.sum(dlambda*wavelength**2*bandpass*vega)/np.sum(dlambda*bandpass*wavelength*vega)
    
    return wl


def calc_synth_phot(wavelength, flux, bandpass):
    dlambda = np.diff(wavelength)
    dlambda = np.concatenate([dlambda, np.array([dlambda[-1]])])

    # assuming a photon-counting device
    phot = np.sum(dlambda*bandpass*wavelength*flux)/np.sum(dlambda*bandpass*wavelength)
    
    return phot


def get_synth_phot(wavelength, flux, scaling=1, band='sdss,g'):
    bandpass = stsyn.band(band)

    wl = calc_pivot_wavelength(wavelength, bandpass(wavelength))
    phot = calc_synth_phot(wavelength, flux, bandpass(wavelength))*scaling
    
    return phot, wl


def get_synth_phot_lco(wavelength, flux, scaling=1, band='SDSS.rp'):
    # Sinistro CCD quantum efficiency
    ccd = ascii.read('../../../../../data/CCD/Sinistro.csv', names=["wavelength", "QE"])
    
    # filter transmission curve
    transmission = ascii.read('../../../../../data/Filters/LCO/' + band + '.txt', names=["wavelength", "transmission"])
    # sort by wavelength (for np.interp)
    transmission.sort('wavelength')    

    # both QE and filter curves' wavelengths are in nm, assuming input wavelength in angstrom
    bandpass = np.interp(wavelength, ccd['wavelength']*10, ccd['QE']/100)*\
                   np.interp(wavelength, transmission['wavelength']*10, transmission['transmission'])
    
    wl = calc_pivot_wavelength(wavelength, bandpass)
    phot = calc_synth_phot(wavelength, flux, bandpass)*scaling
    
    return phot, wl


def get_synth_phot_svo(wavelength, flux, scaling=1, band='GALEX_GALEX.FUV'):
    path = pathlib.Path(__file__).parent.resolve()
    filter_file = os.path.join(path.parents[2], 'data/Filters', band + ".dat")
    # filter transmission curve
    transmission = ascii.read(filter_file, names=["wavelength", "transmission"])
    # sort by wavelength (for np.interp)
    transmission.sort('wavelength')
    
    # assuming wavelength in angstroms
    bandpass = np.interp(wavelength, transmission['wavelength'], transmission['transmission'])
    
    wl = calc_pivot_wavelength(wavelength, bandpass)
    phot = calc_synth_phot(wavelength, flux, bandpass)*scaling
    
    return phot, wl


def apply_extinction(wavelength, flux, ebv=0.1, extinction_model='mwavg'):
    ext = ReddeningLaw.from_extinction_model(extinction_model).extinction_curve(ebv)
    return flux * ext(wavelength)


def deredden(wavelength, flux, ebv=0.1, extinction_model='mwavg'):
    ext = ReddeningLaw.from_extinction_model(extinction_model).extinction_curve(ebv)
    return flux / ext(wavelength)

    
# def blackbody(T, wavelength):
#     bb = 2*const.h*const.c**2/((wavelength**5)*(np.exp(const.h*const.c/(wavelength*const.k_B*T))-1))  # [erg s^-1 cm^-2 cm^-1 sr^-1] intensity
#     bb = bb*np.pi  # [erg s^-1 cm^-2 cm^-1] integrated over half a sphere
#     bb = bb.to(u.erg/u.s/u.cm**2/u.angstrom)
#     return bb
