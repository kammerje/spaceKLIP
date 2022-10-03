from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import os, re, sys
import json

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from astropy.table import Table
from functools import partial
from scipy.integrate import simpson
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter, gaussian_filter
from scipy.optimize import least_squares

import pyklip.klip as klip
import pyklip.instruments.JWST as JWST
import pyklip.fakes as fakes
import pyklip.parallelized as parallelized
import webbpsf_ext

from . import io
from . import utils
from . import plotting

from copy import deepcopy

rad2mas = 180./np.pi*3600.*1000.


# =============================================================================
# MAIN
# =============================================================================

def raw_contrast_curve(meta, fourier=True):
    """
    Compute the raw contrast curves. Known companions and the location of
    the bar mask in both rolls will be masked out.

    Note: currently masks a circle with a radius of 12 FWHM = ~12 lambda/D
          around known companions. This was found to be a sufficient but
          necessary value based on tests with simulated data.

    Note: assumes that the data is photometrically calibrated including pupil
          mask and instrument throughput. Uses an offset PSF from WebbPSF that
          is normalized to a total intensity of 1 to estimate the peak flux of
          a PSF with respect to the source intensity.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    fourier : bool
        Whether to perform shifts in the Fourier plane. This better preserves
        the total flux, however it can introduce Gibbs artefacts for the
        shortest NIRCAM filters as the PSF is undersampled.

    """

    if (meta.verbose == True):
        print('--> Computing raw contrast curve...')

    if (meta.ref_obs is not None) and isinstance(meta.ref_obs, (list,np.ndarray)):
        sci_ref_dir = 'SCI+REF'
    else:
        sci_ref_dir = 'SCI'

    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        if meta.conc_usefile:
            subdir = 'IMGPROCESS/BGSUB'
        elif meta.use_cleaned:
            subdir = f'IMGPROCESS/{sci_ref_dir}_CLEAN'
        else:
            subdir = f'IMGPROCESS/{sci_ref_dir}'
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir=subdir, search=meta.sub_ext)
        meta = utils.prepare_meta(meta, basefiles)
        meta.done_subtraction = True # set the subtraction flag for the subsequent pipeline stages


    # Loop through all directories of subtracted images.
    for counter, rdir in enumerate(meta.rundirs):
        if (meta.verbose == True):
            dirparts = rdir.split('/')[-2].split('_') # -2 because of trailing '/'
            print('--> Mode = {}, annuli = {}, subsections = {}, scenario {} of {}'.format(dirparts[3], dirparts[4], dirparts[5], counter+1, len(meta.rundirs)))

        # Define the input and output directories for each set of pyKLIP
        # parameters.
        idir = rdir+'SUBTRACTED/'
        odir = rdir + 'CONTRAST_RAW/'
        if (not os.path.exists(odir)):
            os.makedirs(odir)

        # Loop through all concatenations.
        for i, key in enumerate(meta.obs.keys()):
            hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
            data = hdul[0].data
            cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
            filt = meta.filter[key]
            mask = meta.coronmsk[key]
            pxsc = meta.pixscale[key] # mas
            pxar = meta.pixar_sr[key] # sr
            wave = meta.wave[filt] # m
            weff = meta.weff[filt] # m
            fwhm = wave/meta.diam*utils.rad2mas/pxsc # pix 1 lambda/D
            if hasattr(meta, 'conc_res'):
                if meta.conc_res == 'default':
                    conc_res = fwhm
                elif isinstance(meta.conc_res, (int, float)):
                    conc_res = meta.conc_res
                else:
                    raise ValueError("Contrast curve resolution must be 'default', or a specific value")
            else:
                conc_res = fwhm

            if hasattr(meta, 'blur_images'):
                if meta.blur_images != False:
                    conc_res += meta.blur_images / 2.355 

            head = hdul[0].header
            hdul.close()

            # Mask out known companions and the location of the bar mask in
            # both rolls.
            if hasattr(meta, 'fwhm_scale'):
                fwhm_scale = meta.fwhm_scale
            else:
                fwhm_scale = 1

            data_masked = data.copy()
            if hasattr(meta, 'ra_off') and hasattr(meta, 'de_off'):
                mask_ra, mask_de = meta.ra_off.copy(), meta.de_off.copy()
                if hasattr(meta, 'ra_off_mask') and hasattr(meta, 'de_off_mask'):
                    mask_ra += meta.ra_off_mask
                    mask_de += meta.de_off_mask
                if isinstance(fwhm_scale, (int,float)):
                    mrads = [fwhm_scale*fwhm]*len(mask_ra)
                elif isinstance(fwhm_scale, (list)):
                    mrads = [fsc*fwhm for fsc in fwhm_scale]
                if len(mrads) != len(mask_ra):
                    raise ValueError('fwhm_scale does not match length of masks')
                data_masked = mask_companions(data, pxsc, cent, mrads, mask_ra, mask_de)

            if (('LWB' in mask) or ('SWB' in mask)):
                data_masked = mask_bar(data_masked, cent, meta.pa_ranges_bar)
            if hasattr(meta, 'pa_ranges_fqpm'):
                data_masked = mask_bar(data_masked, cent, meta.pa_ranges_fqpm)

            if (meta.plotting == True):
                savefile = odir+key+'-mask.pdf'
                plotting.plot_contrast_images(meta, data, data_masked, pxsc, savefile=savefile)

            # Convert the units and compute the contrast. Use the peak pixel
            # count of the recentered offset PSF (discussed with Jason Wang on
            # 12 May 2022).
            offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=True,
                                            derotate=True, fourier=fourier)
            Fstar = meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
            Fdata = data_masked*pxar # MJy; convert the data from MJy/sr to MJy
            seps = [] # arcsec
            cons = []

            try:
                offsetpsf = pyfits.getdata(meta.TA_file, 'SCI')
                offsetpsf -= np.nanmedian(offsetpsf)
                shift = utils.recenter(offsetpsf)
                offsetpsf = utils.fourier_imshift(offsetpsf, shift)
                if ('ND' in pyfits.getheader(meta.TA_file, 0)['SUBARRAY']):
                    wd, od = webbpsf_ext.bandpasses.nircam_com_nd()
                    od_interp = interp1d(wd*1e-6, od)
                    nodes = np.linspace(wave-weff/2., wave+weff/2., 1000)
                    odens = simpson(10**od_interp(nodes), nodes)/weff
                    peak = np.max(offsetpsf)*odens
                else:
                    peak = np.max(offsetpsf)
                TA_filt = pyfits.getheader(meta.TA_file, 0)['FILTER']
                if (TA_filt != filt):
                    peak *= 10**(-(meta.mstar[filt]-meta.mstar[TA_filt])/2.5)
                for j in range(data_masked.shape[0]):
                    sep, con = klip.meas_contrast(dat=data_masked[j]/peak, iwa=meta.iwa, owa=meta.owa, resolution=conc_res, center=cent, low_pass_filter=False)
                    seps += [sep*pxsc/1000.] # arcsec
                    cons += [con]
            except: 
                offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=True, derotate=False)
                Fstar = meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                Fdata = data_masked*pxar # MJy; convert the data from MJy/sr to MJy
                for j in range(Fdata.shape[0]):
                    #IWA is already baked in to the subtraction as NaN pixels, so set to 0
                    sep, con = klip.meas_contrast(dat=Fdata[j]/Fstar, iwa=0, owa=meta.owa, resolution=conc_res, center=cent, low_pass_filter=False)
                    seps += [sep*pxsc/1000.] # arcsec
                    cons += [con]

            # Save the contrast curve as a dictionary
            save_dict = {'seps':seps[0].tolist(), 'cons':{}, 'mstar':meta.mstar}
            for j, con in enumerate(cons):
                save_dict['cons']['KL{}'.format(meta.numbasis[j])] = cons[j].tolist()
            rawconfile = odir+key+'-raw_save.json'
            with open(rawconfile, 'w') as rf:
                json.dump(save_dict, rf)

            # Plot a figure of the contrast curves
            if (meta.plotting == True):
                savefile = odir+key+'-cons_raw.pdf'
                labels = []
                # for j in range(Fdata.shape[0]):
                for j in range(data_masked.shape[0]):
                    labels.append(str(head['KLMODE{}'.format(j)])+' KL')
                plotting.plot_contrast_raw(meta, seps[0], cons, labels=labels, savefile=savefile)


    return None

def calibrated_contrast_curve(meta, fourier=False):
    """
    Compute the calibrated contrast curves. Injection and recovery tests
    are performed to estimate the algo & coronmsk throughput.

    Note: do not inject fake planets on top of the bar mask in either of
          the rolls!

    Note: currently forces any fake companions to be injected at least 10
          FWHM = ~10 lambda/D away from other fake companions or known
          companions. This was found to be a sufficient but necessary
          value based on tests with simulated data.

    TODO: use a position dependent offset PSF from pyNRC instead of the
          completely unocculted offset PSF from WebbPSF.

    Parameters
    ----------
    mstar: dict of float
        Host star magnitude in each filter. Must contain one entry for
        each filter used in the data in the input directory.
    ra_off: list of float
        RA offset of the known companions.
    de_off: list of float
        DEC offset of the known companions.
    seps_inject_rnd: list of float
        List of separations at which fake planets shall be injected to
        compute the calibrated contrast curve for the round masks.
    pas_inject_rnd: list of float
        List of position angles at which fake planets shall be injected to
        compute the calibrated contrast curve for the round masks.
    seps_inject_bar: list of float
        List of separations at which fake planets shall be injected to
        compute the calibrated contrast curve for the bar masks.
    pas_inject_bar: list of float
        List of position angles at which fake planets shall be injected to
        compute the calibrated contrast curve for the bar masks.
    KL: int
        Index of the KL component for which the calibrated contrast curve
        and the companion properties shall be computed.
    overwrite: bool
        If true overwrite existing data.
    fourier : bool
        Whether to perform shifts in the Fourier plane. This better preserves
        the total flux, however it can introduce Gibbs artefacts for the
        shortest NIRCAM filters as the PSF is undersampled.
    """
    if (meta.ref_obs is not None) and isinstance(meta.ref_obs, (list,np.ndarray)):
        sci_ref_dir = 'SCI+REF'
    else:
        sci_ref_dir = 'SCI'

    # If necessary, build the obs dictionary etc
    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        if meta.conc_usefile == 'bgsub':
            subdir = 'IMGPROCESS/BGSUB'
        elif meta.use_cleaned:
            subdir = f'IMGPROCESS/{sci_ref_dir}_CLEAN'
        else:
            subdir = f'IMGPROCESS/{sci_ref_dir}'
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir=subdir, search=meta.sub_ext)
        meta = utils.prepare_meta(meta, basefiles)
        meta.done_subtraction = True # set the subtraction flag for the subsequent pipeline stages


    if (meta.verbose == True):
        print('--> Computing calibrated contrast curve...')

    # Make inputs arrays.
    seps_inject_rnd = np.array(meta.seps_inject_rnd)
    pas_inject_rnd = np.array(meta.pas_inject_rnd)
    seps_inject_bar = np.array(meta.seps_inject_bar)
    pas_inject_bar = np.array(meta.pas_inject_bar)
    if hasattr(meta, 'seps_inject_fqpm'):
        seps_inject_fqpm = np.array(meta.seps_inject_fqpm)
    if hasattr(meta, 'pas_inject_fqpm'):
        pas_inject_fqpm = np.array(meta.pas_inject_fqpm)

    meta.truenumbasis = {}
    for counter, rdir in enumerate(meta.rundirs):
        # Check if run directory actually exists
        if not os.path.exists(rdir):
            raise ValueError('Could not find provided run directory "{}"'.format(rdir))

        # Get some information from the original meta file in the run directory
        metasave = io.read_metajson(rdir+'SUBTRACTED/MetaSave.json')
        mode = metasave['used_mode']
        annuli = metasave['used_annuli']
        subsections = metasave['used_subsections']

        if (meta.verbose == True):
            sys.stdout.write('\r--> Mode = {}, annuli = {}, subsections = {}, scenario {} of {}'.format(mode, annuli, subsections, counter+1, len(meta.rundirs)))
            sys.stdout.flush()

        # Define the input and output directories for each set of pyKLIP parameters.
        idir = rdir + 'SUBTRACTED/'
        odir = rdir + 'CONTRAST_KL{}/'.format(meta.KL)
        if (not os.path.exists(odir)):
            os.makedirs(odir)

        # Loop through all sets of observing parameters.
        for i, key in enumerate(meta.obs.keys()):

            ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]
            filepaths = np.array(meta.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
            ww_cal = np.where(meta.obs[key]['TYP'] == 'CAL')[0]
            psflib_filepaths = np.array(meta.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
            hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
            data = hdul[0].data
            pxsc = meta.obs[key]['PIXSCALE'][0] # mas
            cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
            temp = [s.start() for s in re.finditer('_', key)]
            inst = key[:temp[0]]
            filt = key[temp[1]+1:temp[2]]
            mask = key[temp[3]+1:temp[4]]
            subarr = key[temp[4]+1:]
            wave = meta.wave[filt] # m
            fwhm = wave/meta.diam*utils.rad2mas/pxsc # pix

            # Define contrast resolution which dictates the initial guess
            # for the FWHM of the gaussian we fit to the PSF when recovering
            # injected planet flux
            if hasattr(meta, 'conc_res'):
                if meta.conc_res == 'default':
                    conc_res = fwhm
                elif isinstance(meta.conc_res, (int, float)):
                    conc_res = meta.conc_res
                else:
                    raise ValueError("Contrast curve resolution must be 'default', or a specific value")

            all_numbasis = []
            # Loop over up to 100 different KL mode inputs
            for i in range(100):
                try:
                    # Get value from header
                    all_numbasis.append(hdul[0].header['KLMODE{}'.format(i)])
                except:
                    # No more KL modes
                    continue

            meta.truenumbasis[key] = [num for num in all_numbasis if (num <= meta.maxnumbasis[key])]
            
            if meta.KL == 'max':
                if '_ADI_' in rdir:
                    KL = meta.adimaxnumbasis[key]
                elif '_RDI_' in rdir:
                    KL = meta.rdimaxnumbasis[key]
                elif '_ADI+RDI_' in rdir:
                    KL = meta.maxnumbasis[key]
            else:
                KL = meta.KL

            # Get the index of the KL component we are interested in
            try:
                KLindex = all_numbasis.index(KL)
            except:
                raise ValueError('KL={} not found. Calculated options are: {}, and maximum possible for this data is {}'.format(KL, all_numbasis, meta.maxnumbasis[key]))

            hdul.close()

            # Load raw contrast curves. If overwrite is false,
            # check whether the calibrated contrast curves have
            # been computed already.
            rawdir = '/'.join(odir.split('/')[:-2])+'/CONTRAST_RAW/'
            rawconfile = rawdir+key+'-raw_save.json'
            with open(rawconfile, 'r') as rf:
                rawcon = json.load(rf)

            seps = rawcon['seps']
            cons = rawcon['cons']['KL{}'.format(KL)]

            if meta.overwrite == False:
                try:
                    calconfile = odir+key+'-cal_save.json'
                    with open(calconfile, 'r') as rf:
                        calcon = json.load(rf)
                    flux_all = calcon['flux_all']
                    seps_all = calcon['seps_all']
                    pas_all = calcon['pas_all']
                    flux_retr_all = calcon['flux_retr_all']
                    todo = False
                except:
                    todo = True
            else:
                todo = True

            # 2D map of the total throughput, i.e., an integration
            # time weighted average of the coronmsk transmission
            # over the rolls.
            tottp = utils.get_transmission(meta, key, odir)

            # The calibrated contrast curves have not been
            # computed already.
            if (todo == True):

                # Offset PSF from WebbPSF, i.e., an integration
                # time weighted average of the unocculted offset
                # PSF over the rolls (does account for pupil mask
                # throughput).
                offsetpsf = utils.get_offsetpsf(meta, key,
                                                recenter_offsetpsf=False,
                                                derotate=True, fourier=fourier)

                # Convert the units and compute the injected
                # fluxes. They need to be in the units of the data
                # which is MJy/sr.
                Fstar = meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                con_interp = interp1d(seps, cons)
                if (mask in ['MASKASWB', 'MASKALWB']):
                    cons_inject = con_interp(seps_inject_bar*pxsc/1000)*20.
                elif ('4QPM' in mask):
                    cons_inject = con_interp(seps_inject_fqpm*pxsc/1000.)*20.
                else:
                    cons_inject = con_interp(seps_inject_rnd*pxsc/1000.)*20. # inject at 100 sigma
                flux_inject = cons_inject*Fstar*(180./np.pi*3600.*1000.)**2/pxsc**2 # MJy/sr; convert the injected flux from contrast to MJy/sr

                # Assemble mask regions
                if hasattr(meta, 'fwhm_scale'):
                    fwhm_scale = meta.fwhm_scale
                else:
                    fwhm_scale = 1

                if hasattr(meta, 'ra_off') and hasattr(meta, 'de_off'):
                    mask_ra, mask_de = meta.ra_off.copy(), meta.de_off.copy()
                if hasattr(meta, 'ra_off_mask') and hasattr(meta, 'de_off_mask'):
                    mask_ra += meta.ra_off_mask
                    mask_de += meta.de_off_mask
                if isinstance(fwhm_scale, (int,float)):
                    mrads = [fwhm_scale*fwhm]*len(mask_ra)
                elif isinstance(fwhm_scale, (list)):
                    mrads = [fsc*fwhm for fsc in fwhm_scale]
                if len(mrads) != len(mask_ra):
                    raise ValueError('fwhm_scale does not match length of masks')

                # If the separation is too small the
                # interpolation of the contrasts returns nans,
                # these need to be filtered out before feeding the
                # fluxes into the injection and recovery routine.
                good = np.isnan(flux_inject) == False
                if (mask in ['MASKASWB', 'MASKALWB']):
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, conc_res, mrads, flux_inject[good], seps_inject_bar[good], pas_inject_bar, KLindex, mask_ra, mask_de, fourier)
                elif ('4QPM' in mask):
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, conc_res, mrads, flux_inject[good], seps_inject_fqpm[good], pas_inject_fqpm, KLindex, mask_ra, mask_de, fourier)
                else:
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, conc_res, mrads, flux_inject[good], seps_inject_rnd[good], pas_inject_rnd, KLindex, mask_ra, mask_de, fourier)
            
            # Decide how to calibrate the contrast curve
            if hasattr(meta, 'cal_method'):
                method = meta.cal_method
            else:
                method = 'model'

            # Want to add a point at 0 throughput, 0 separation
            flux_tab = np.append([1],flux_all)
            seps_tab = np.append([0],seps_all)
            pas_tab = np.append([0],pas_all)
            flux_retr_tab = np.append([1e-6],flux_retr_all)

            # Group the injection and recovery results by separation and take the median
            res = Table([flux_tab, seps_tab, pas_tab, flux_retr_tab], names=('flux', 'seps', 'pas', 'flux_retr'))
            res['tps'] = res['flux_retr']/res['flux']
            med_res = res.group_by('seps')
            med_res = med_res.groups.aggregate(np.nanmedian)
            
            # Decide how to calibrate the contrast curve
            if hasattr(meta, 'cal_method'):
                method = meta.cal_method
            else:
                method = 'model'
            if method == 'model':
                # Use a fit to a logistic growth function to them.
                p0 = np.array([1., 0., 1., 0.2, 15.])
                pp = least_squares(func_lnprob, p0, args=(med_res['seps'], med_res['tps']))
                # p0 = np.array([0., 1., 1.])
                # pp = least_squares(self.growth_lnprob, p0, args=(med_res['seps']*pxsc/1000., med_res['tps']))
                corr = func(pp['x'], np.array(seps)*1000./pxsc)
            elif method == 'median':
                # Interpolate over the median
                med_interp = interp1d(med_res['seps'], med_res['tps'], fill_value=(1e-10,med_res['tps'][-1]), \
                                        bounds_error=False)
                corr = med_interp(np.array(seps)*1000./pxsc)

            # Calibrate contrast
            corr_cons = np.array(cons) / corr

            # Save the contrast curve and other properties as a dictionary
            save_dict = {'seps_all':seps_all.tolist(), 'flux_all':flux_all.tolist(), 'pas_all':pas_all.tolist(), 'flux_retr_all':flux_retr_all.tolist(), 'seps':seps, 'corr_cons':corr_cons.tolist(), 'raw_cons':cons}
            calconfile = odir+key+'-cal_save.json'
            with open(calconfile, 'w') as cf:
                json.dump(save_dict, cf)

            if meta.plotting:
                # Plot injected locations
                savefile=odir+key+'-cons_inj.pdf'
                plotting.plot_injected_locs(meta, data[KLindex], tottp, seps_all, pas_all, pxsc=pxsc, savefile=savefile)

                # Plot calibrated contrast
                fit_thrput = {}
                if (mask in ['MASKASWB', 'MASKALWB']):
                    fit_thrput['seps'] = np.linspace(seps_inject_bar[0], seps_inject_bar[-1], 100)
                else:
                    fit_thrput['seps'] = np.linspace(0, seps_inject_rnd[-1], 100)

                if method == 'model':
                    fit_thrput['tps'] = func(pp['x'], fit_thrput['seps'])
                elif method == 'median':
                    fit_thrput['tps'] = med_interp(fit_thrput['seps'])

                savefile=odir+key+'-cons_cal.pdf'
                plotting.plot_contrast_calibrated(res, med_res, fit_thrput, seps, cons, corr_cons, savefile=savefile)

    if (meta.verbose == True):
        print('')

    return None


def mask_companions(data, pxsc, cent, mrads, ra_off=[], de_off=[]):
    """
    Mask out known companions.

    Parameters
    ----------
    data: array
        Data cube of shape (Nframes, Npix, Npix) in which the companions
        shall be masked out.
    pxsc: float
        Pixel scale of the data.
    cent: tuple of float
        Center of the data (i.e., center of the host star PSF).
    mrad: list
        Radii of the masks that are used to mask out the companions.
    ra_off: list of float
        RA offset of the companions that shall be masked out.
    de_off: list of float
        DEC offset of the companions that shall be masked out.

    Returns
    -------
    data_masked: array
        Data cube of shape (Nframes, Npix, Npix) in which the companions
        are masked out.
    """

    # Sanitize inputs.
    if (len(data.shape) != 3):
        raise UserWarning('Data has invalid shape')

    # Mask out known companions.
    data_masked = data.copy()
    yy, xx = np.indices(data.shape[1:]) # pix
    for i, loc in enumerate(zip(ra_off, de_off)):
        ra, de = loc
        dist = np.sqrt((xx-cent[0]+ra/pxsc)**2+(yy-cent[1]-de/pxsc)**2) # pix
        data_masked[:, dist <= mrads[i]] = np.nan

    return data_masked


def mask_bar(data, cent, pa_ranges_bar=[]):
    """
    Mask out bar mask occulter. This is done by specifying pizza slices
    that shall be considered when computing the contrast curves.

    Note: make sure to mask out the bar mask in either of the rolls!

    Parameters
    ----------
    data: array
        Data cube of shape (Nframes, Npix, Npix) in which the bar mask
        occulter shall be masked out.
    cent: tuple of float
        Center of the data (i.e., center of the host star PSF) (pixels).
    pa_ranges_bar: list of tuple of float
        List of tuples defining the pizza slices that shall be considered
        when computing the contrast curves for the bar masks (deg).

    Returns
    -------
    data_masked: array
        Data cube of shape (Nframes, Npix, Npix) in which everything but
        the specified pizza slices is masked out.
    """

    # Sanitize inputs.
    if (len(data.shape) != 3):
        raise UserWarning('Data has invalid shape')

    # Mask out bar mask occulter.
    data_masked = data.copy()
    yy, xx = np.indices(data.shape[1:]) # pix
    tt = np.rad2deg(-1.*np.arctan2((xx-cent[0]), (yy-cent[1]))) % 360. # deg

    for i in range(len(pa_ranges_bar)):
        if (i == 0):
            data_masked[:] = np.nan
        mask = (tt >= pa_ranges_bar[i][0]) & (tt <= pa_ranges_bar[i][1])
        data_masked[:, mask] = data[:, mask]

    return data_masked

def inject_recover(meta,
                   filepaths,
                   psflib_filepaths,
                   mode,
                   odir,
                   key,
                   annuli,
                   subsections,
                   pxsc, # mas
                   inst,
                   filt,
                   mask,
                   fwhm, 
                   mrad, # pix
                   flux_inject=[], # MJy/sr
                   seps_inject=[], # pix
                   pas_inject=[], # deg
                   KL=-1,
                   ra_off=[], # mas
                   de_off=[],
                   fourier=True): # mas
    """
    Inject fake companions and recover them by fitting a 2D Gaussian.
    Makes sure that the injected fake companions are not too close to any
    real companions and also that fake companions which are too close to
    each other are injected into different fake datasets.

    TODO: recover fake companions by fitting an offset PSF instead of a 2D
          Gaussian.

    TODO: currently uses WebbPSF to compute a theoretical offset PSF. It
          should be possible to use PSF stamps extracted from the
          astrometric confirmation images in the future.

    TODO: use a position dependent offset PSF from pyNRC instead of the
          completely unocculted offset PSF from WebbPSF.

    Parameters
    ----------
    filepaths: list of str
        List of stage 2 reduced science target files for the considered
        observation.
    psflib_filepaths: list of str
        List of stage 2 reduced reference target files for the considered
        observation.
    mode: list of str
        List of modes for pyKLIP, will loop through all.
    odir: str
        Output directory for the plots and pyKLIP products.
    key: str
        Dictionary key of the self.obs dictionary specifying the
        considered observation.
    annuli: list of int
        List of number of annuli for pyKLIP, will loop through all.
    subsections: list of int
        List of number of subsections for pyKLIP, will loop through all.
    pxsc: float
        Pixel scale of the data.
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    mrad: float
        Radius of the mask that is used to mask out the companions.
    flux_inject: list of float
        List of fluxes at which fake planets shall be injected.
    seps_inject: list of float
        List of separations at which fake planets shall be injected.
    pas_inject: list of float
        List of position angles at which fake planets shall be injected.
    KL: int
        Index of the KL component for which the fake companions shall be
        injected.
    ra_off: list of float
        RA offset of the companions that shall be masked out.
    de_off: list of float
        DEC offset of the companions that shall be masked out.
    fourier : bool
        Whether to perform shifts in the Fourier plane. This better preserves
        the total flux, however it can introduce Gibbs artefacts for the
        shortest NIRCAM filters as the PSF is undersampled.

    Returns
    -------

    """

    # Sanitize inputs.
    Nfl = len(flux_inject)
    Nse = len(seps_inject)
    Npa = len(pas_inject)
    if (Nfl != Nse):
        raise UserWarning('Injected fluxes need to match injected separations')

    # Create an output directory for the datasets with fake companions.
    odir_temp = odir+'INJECTED/'
    if (not os.path.exists(odir_temp)):
        os.makedirs(odir_temp)

    # Initialize outputs.
    flux_all = [] # MJy/sr
    seps_all = [] # pix
    pas_all = [] # deg
    flux_retr_all = [] # MJy/sr
    done = []

    # Do not inject fake companions closer than mrad to any known
    # companion.
    if isinstance(mrad, list):
        # Set mrad to whatever the first mrad is in the list
        mrad = mrad[0]
    for i in range(Nse):
        for j in range(Npa):
            ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
            de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
            for k in range(len(ra_off)):
                dist = np.sqrt((ra-ra_off[k])**2+(de-de_off[k])**2) # mas
                #Want to lose injections close to masked regions but 
                #keep the injections that are close to the center
                if (dist < mrad*pxsc) and (seps_inject[i] > 10):
                    done += [i*Npa+j]
                    break

    if not hasattr(meta, 'blur_images'):
        meta.blur_images = False

    if not hasattr(meta, 'repeatcentering_companion'):
        centering_alg = 'savefile'
    else:
        if meta.repeatcentering_companion == False:
            centering_alg = 'savefile'
        else:
            centering_alg = meta.repeatcentering_companion

    # Offset PSF from WebbPSF
    offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=False,
                                    derotate=False, fourier=fourier)
    offsetpsf /= np.max(offsetpsf)

    load_file0_center = meta.load_file0_center if hasattr(meta,'load_file0_center') else False
    # If not finished yet, create a new pyKLIP dataset into which fake
    # companions will be injected.
    raw_dataset = JWST.JWSTData(filepaths=filepaths,
                        psflib_filepaths=psflib_filepaths, centering=centering_alg, badpix_threshold=meta.badpix_threshold,
                        scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts', spectral_type=meta.spt, load_file0_center=load_file0_center,
                        save_center_file=meta.ancildir+'shifts/file0_centers',
                        fiducial_point_override=meta.fiducial_point_override, blur=meta.blur_images)
    finished = False
    ctr = 0
    while (finished == False):
        # Make a copy of the dataset so we don't overwrite things
        dataset = deepcopy(raw_dataset)

        # Inject fake companions. Make sure that no other fake companion
        # closer than mrad will be injected into the same dataset.
        injected = []
        breakflag = False
        # Loop over separations
        for i in range(Nse):
            # Loop over PA's
            for j in range(Npa):
                # Check if this Sep/PA has already been injected
                if (i*Npa+j not in done):
                    # Convert to RA/Dec
                    ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
                    de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
                    flag = True # Flag for injection
                    # Compare RA/Dec to already injected objects
                    for k in range(len(injected)):
                        jj = injected[k] % Npa
                        ii = (injected[k]-jj)//Npa
                        ra_temp = seps_inject[ii]*pxsc*np.sin(np.deg2rad(pas_inject[jj])) # mas
                        de_temp = seps_inject[ii]*pxsc*np.cos(np.deg2rad(pas_inject[jj])) # mas
                        dist = np.sqrt((ra-ra_temp)**2+(de-de_temp)**2) # mas
                        # If object is too close to an already injected object, do *not* inject.
                        # If previous object is too close to the coronagraph, also don't inject 
                        # any as this can really affect the KLIP subtraction
                        if (dist < 2* mrad*pxsc) or (seps_inject[ii] < 5):
                            flag = False
                    # If flagged for injection, then inject the planet.
                    if (flag == True):
                        injected += [i*Npa+j] #Mark as injected in this dataset
                        done += [i*Npa+j]   # Mark as 'done' (i.e. injected in any dataset)
                        if meta.blur_images == False:
                            stamp = np.array([offsetpsf*flux_inject[i] for k in range(dataset.input.shape[0])]) # MJy/sr
                        else:
                            # If the image has been blurred, inject Gaussians instead. 
                            # pyKLIP can't shift an undersampled PSF without artifacts, so there isn't currently
                            # a method to use the instrumental PSF. 
                            stamp = np.array([flux_inject[i] for k in range(dataset.input.shape[0])]) # MJy/sr
                        fakes.inject_planet(frames=dataset.input, centers=dataset.mask_centers, inputflux=stamp, astr_hdrs=dataset.wcs, radius=seps_inject[i], pa=pas_inject[j], field_dependent_correction=partial(utils.field_dependent_correction, meta=meta))
                        flux_all += [flux_inject[i]] # MJy/sr
                        seps_all += [seps_inject[i]] # pix
                        pas_all += [pas_inject[j]] # deg

        # Sometimes faster to just use 1 thread. 
        cal_nthreads = getattr(meta, 'cal_nthreads', None)

        # Run pyKLIP.
        parallelized.klip_dataset(dataset=dataset,
                                  mode=mode,
                                  outputdir=odir_temp,
                                  fileprefix='FAKE_%04.0f_' % ctr+key,
                                  annuli=annuli,
                                  subsections=subsections,
                                  movement=1,
                                  numbasis=[meta.truenumbasis[key][KL]],
                                  calibrate_flux=False,
                                  numthreads=cal_nthreads,
                                  maxnumbasis=meta.maxnumbasis[key],
                                  psf_library=dataset.psflib,
                                  highpass=False,
                                  verbose=False)

        # Recover fake companions by fitting a 2D Gaussian, mainly interested in peak flux
        # so this is a reasonable approximation. 
        klipdataset = odir_temp+'FAKE_%04.0f_' % ctr+key+'-KLmodes-all.fits'
        with pyfits.open(klipdataset) as hdul:
            outputfile = hdul[0].data[0]
            outputfile_centers = [hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']]
        for k in range(len(injected)):
            jj = injected[k] % Npa
            ii = (injected[k]-jj)//Npa
            fake_flux = fakes.retrieve_planet_flux(frames=outputfile, centers=outputfile_centers, astr_hdrs=dataset.output_wcs[0], sep=seps_inject[ii], pa=pas_inject[jj], searchrad=5, guessfwhm=fwhm, refinefit=True) #Refine fit makes a big difference
            if fake_flux < 0:
                fake_flux = 0 #Negative flux = zero flux
            flux_retr_all += [fake_flux] # MJy/sr
        ctr += 1

        # Check if finished, i.e., if all fake companions were injected.
        # If not, continue with a new dataset.
        if (len(done) == Nse*Npa):
            finished = True

    # Make outputs arrays.
    flux_all = np.array(flux_all) # MJy/sr
    seps_all = np.array(seps_all) # pix
    pas_all = np.array(pas_all) # deg
    flux_retr_all = np.array(flux_retr_all) # MJy/sr

    return flux_all, seps_all, pas_all, flux_retr_all


def func(p,x):

    y = p[0]*(1.-np.exp(-(x-p[1])**2/(2*p[2]**2)))*(1-p[3]*np.exp(-(x-p[4])**2/(2*p[2]**2)))
    y[x < p[1]] = 1e-10

    return y

def func_lnprob(p,x,y):
    return np.abs(y[:-1]-func(p, x[:-1]))
