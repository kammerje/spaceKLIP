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
from scipy.optimize import least_squares

import pyklip.klip as klip
import pyklip.instruments.JWST as JWST
import pyklip.fakes as fakes
import pyklip.parallelized as parallelized

from . import io
from . import utils
from . import plotting

rad2mas = 180./np.pi*3600.*1000.


# =============================================================================
# MAIN
# =============================================================================

def raw_contrast_curve(meta):
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
    
    """
    
    if (meta.verbose == True):
        print('--> Computing raw contrast curve...')
    
    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS', search=meta.sub_ext)
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
        odir = rdir+'CONTRAST/'
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
            fwhm = wave/meta.diam*utils.rad2mas/pxsc # pix
            head = hdul[0].header
            hdul.close()
            
            # Mask out known companions and the location of the bar mask in
            # both rolls.
            if ('4QPM' in mask):
                fwhm_scale = 4.
            else:
                fwhm_scale = 12.
            data_masked = mask_companions(data, pxsc, cent, fwhm_scale*fwhm, meta.ra_off, meta.de_off)

            if (('LWB' in mask) or ('SWB' in mask)):
                data_masked = mask_bar(data_masked, cent, meta.pa_ranges_bar)
            
            if (meta.plotting == True):
                savefile = odir+key+'-mask.pdf'
                plotting.plot_contrast_images(meta, data, data_masked, pxsc, savefile=savefile)
            
            # Convert the units and compute the contrast. Use the peak pixel
            # count of the recentered offset PSF (discussed with Jason Wang on
            # 12 May 2022).
            offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=True, derotate=True)
            Fstar = meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
            Fdata = data_masked*pxar # MJy; convert the data from MJy/sr to MJy
            seps = [] # arcsec
            cons = []
            for j in range(Fdata.shape[0]):
                sep, con = klip.meas_contrast(dat=Fdata[j]/Fstar, iwa=meta.iwa, owa=meta.owa, resolution=2.*fwhm, center=cent, low_pass_filter=False)
                seps += [sep*pxsc/1000.] # arcsec
                cons += [con]
            # seps = np.array(seps) # arcsec
            # cons = np.array(cons)

            # Save the contrast curve as a dictionary
            save_dict = {'seps':seps[0].tolist(), 'cons':{}}
            for j, con in enumerate(cons):
                save_dict['cons']['KL{}'.format(meta.numbasis[j])] = cons[j].tolist()
            rawconfile = odir+key+'-raw_save.json'
            with open(rawconfile, 'w') as rf:
                json.dump(save_dict, rf)

            # np.save(odir+key+'-seps.npy', seps) # arcsec
            # np.save(odir+key+'-cons.npy', cons)
            
            if (meta.plotting == True):
                savefile = odir+key+'-cons_raw.pdf'
                labels = []
                for j in range(Fdata.shape[0]):
                    labels.append(str(head['KLMODE{}'.format(j)])+' KL')
                plotting.plot_contrast_raw(meta, seps[0], cons, labels=labels, savefile=savefile)

    
    return None

def calibrated_contrast_curve(meta):
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
    """
    # If necessary, build the obs dictionary etc
    if not meta.done_subtraction:
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS', search=meta.sub_ext)
        meta = utils.prepare_meta(meta, basefiles)
        # Set the subtraction flag for other stages
        meta.done_subtraction = True

    if (meta.verbose == True):
        print('--> Computing calibrated contrast curve...')
    
    # Make inputs arrays.
    seps_inject_rnd = np.array(meta.seps_inject_rnd)
    pas_inject_rnd = np.array(meta.pas_inject_rnd)
    seps_inject_bar = np.array(meta.seps_inject_bar)
    pas_inject_bar = np.array(meta.pas_inject_bar)
    seps_inject_fqpm = np.array(meta.seps_inject_fqpm)
    pas_inject_fqpm = np.array(meta.pas_inject_fqpm)
    
    # Loop through all modes, numbers of annuli, and numbers of
    # subsections.
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
        odir = rdir + 'CONTRAST/'
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
            # Get the index of the KL component we are interested in
            KLindex = all_numbasis.index(meta.KL)

            hdul.close()
            
            # Load raw contrast curves. If overwrite is false,
            # check whether the calibrated contrast curves have
            # been computed already.
            rawconfile = odir+key+'-raw_save.json'
            with open(rawconfile, 'r') as rf:
                rawcon = json.load(rf)

            seps = rawcon['seps']
            cons = rawcon['cons']['KL{}'.format(meta.KL)]

            # seps = np.load(odir+key+'-seps.npy')[meta.KL] # arcsec
            # cons = np.load(odir+key+'-cons.npy')[meta.KL]
            if meta.overwrite == False:
                try:
                    calconfile = odir+key+'-cal_save.json'
                    with open(calconfile, 'r') as rf:
                        calcon = json.load(rf)

                    flux_all = calcon['flux_all']
                    seps_all = calcon['seps_all']
                    pas_all = calcon['pas_all']
                    flux_retr_all = calcon['flux_retr_all']

                    # flux_all = np.load(odir+key+'-flux_all.npy') # MJy/sr
                    # seps_all = np.load(odir+key+'-seps_all.npy') # pix
                    # pas_all = np.load(odir+key+'-pas_all.npy') # deg
                    # flux_retr_all = np.load(odir+key+'-flux_retr_all.npy') # MJy/sr
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
                offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=False, derotate=True)
                
                # Convert the units and compute the injected
                # fluxes. They need to be in the units of the data
                # which is MJy/sr.
                Fstar = meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                if (mask in ['MASKASWB', 'MASKALWB']):
                    cons_inject = np.interp(seps_inject_bar*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                elif ('4QPM' in mask):
                    cons_inject = np.interp(seps_inject_fqpm*pxsc/1000., seps, cons)*5.
                else:
                    cons_inject = np.interp(seps_inject_rnd*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                flux_inject = cons_inject*Fstar*(180./np.pi*3600.*1000.)**2/pxsc**2 # MJy/sr; convert the injected flux from contrast to MJy/sr
                
                # If the separation is too small the
                # interpolation of the contrasts returns nans,
                # these need to be filtered out before feeding the
                # fluxes into the injection and recovery routine.
                good = np.isnan(flux_inject) == False
                if (mask in ['MASKASWB', 'MASKALWB']):
                    fwhm_scale = 10
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, fwhm_scale*fwhm, flux_inject[good], seps_inject_bar[good], pas_inject_bar, KLindex, meta.ra_off, meta.de_off)
                elif ('4QPM' in mask):
                    fwhm_scale = 4
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, fwhm_scale*fwhm, flux_inject[good], seps_inject_fqpm[good], pas_inject_fqpm, KLindex, meta.ra_off, meta.de_off)
                else:
                    fwhm_scale = 10
                    flux_all, seps_all, pas_all, flux_retr_all = inject_recover(meta, filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, inst, filt, mask, fwhm_scale*fwhm, flux_inject[good], seps_inject_rnd[good], pas_inject_rnd, KLindex, meta.ra_off, meta.de_off)

                # np.save(odir+key+'-flux_all.npy', flux_all) # MJy/sr
                # np.save(odir+key+'-seps_all.npy', seps_all) # pix
                # np.save(odir+key+'-pas_all.npy', pas_all) # deg
                # np.save(odir+key+'-flux_retr_all.npy', flux_retr_all) # MJy/sr
            
            # Group the injection and recovery results by
            # separation and take the median, then fit a logistic
            # growth function to them.
            res = Table([flux_all, seps_all, pas_all, flux_retr_all], names=('flux', 'seps', 'pas', 'flux_retr'))
            res['tps'] = res['flux_retr']/res['flux']
            med_res = res.group_by('seps')
            med_res = med_res.groups.aggregate(np.nanmedian)
            p0 = np.array([1., 0., 1., 0.2, 15.])
            pp = least_squares(func_lnprob, p0, args=(med_res['seps'], med_res['tps']))
            # p0 = np.array([0., 1., 1.])
            # pp = least_squares(self.growth_lnprob, p0, args=(med_res['seps']*pxsc/1000., med_res['tps']))
            corr_cons = np.array(cons)/func(pp['x'], np.array(seps)*1000./pxsc)
            
            # np.save(odir+key+'-pp.npy', pp['x'])

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
                    fit_thrput['seps'] = np.linspace(seps_inject_rnd[0], seps_inject_rnd[-1], 100)

                fit_thrput['tps'] = func(pp['x'], fit_thrput['seps'])

                savefile=odir+key+'-cons_cal.pdf'
                plotting.plot_contrast_calibrated(res, med_res, fit_thrput, seps, cons, corr_cons, savefile=savefile)
            
    if (meta.verbose == True):
        print('')
    
    return None

 
def mask_companions(data, pxsc, cent, mrad, ra_off=[], de_off=[]):
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
    mrad: float
        Radius of the mask that is used to mask out the companions.
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
    for ra, de in zip(ra_off, de_off):
        dist = np.sqrt((xx-cent[0]+ra/pxsc)**2+(yy-cent[1]-de/pxsc)**2) # pix
        data_masked[:, dist <= mrad] = np.nan
    
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


def inject_recover_advanced(meta,
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
                           mrad, # pix
                           flux_inject=[], # MJy/sr
                           seps_inject=[], # pix
                           pas_inject=[], # deg
                           KL=-1,
                           ra_off=[], # mas
                           de_off=[]): # mas

    # TODO Create injection where PSFs are generated exactly at the location they should be injected

    return

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
                   mrad, # pix
                   flux_inject=[], # MJy/sr
                   seps_inject=[], # pix
                   pas_inject=[], # deg
                   KL=-1,
                   ra_off=[], # mas
                   de_off=[]): # mas
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
    
    # Offset PSF from WebbPSF, i.e., an integration time weighted average
    # of the unocculted offset PSF over the rolls (normalized to a peak
    # flux of 1).
    offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=False, derotate=True)
    offsetpsf /= np.max(offsetpsf)
    
    # Initialize outputs.
    flux_all = [] # MJy/sr
    seps_all = [] # pix
    pas_all = [] # deg
    flux_retr_all = [] # MJy/sr
    done = []
    
    # Do not inject fake companions closer than mrad to any known
    # companion.
    for i in range(Nse):
        for j in range(Npa):
            ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
            de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
            for k in range(len(ra_off)):
                dist = np.sqrt((ra-ra_off[k])**2+(de-de_off[k])**2) # mas
                if (dist < mrad*pxsc):
                    done += [i*Npa+j]
                    break
    
    # If not finished yet, create a new pyKLIP dataset into which fake
    # companions will be injected.
    finished = False
    ctr = 0
    while (finished == False):
        dataset = JWST.JWSTData(filepaths=filepaths,
                                psflib_filepaths=psflib_filepaths, centering=meta.centering_alg)
        
        # Inject fake companions. Make sure that no other fake companion
        # closer than mrad will be injected into the same dataset.
        todo = []
        for i in range(Nse):
            for j in range(Npa):
                if (i*Npa+j not in done):
                    ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
                    de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
                    flag = True
                    for k in range(len(todo)):
                        jj = todo[k] % Npa
                        ii = (todo[k]-jj)//Npa
                        ra_temp = seps_inject[ii]*pxsc*np.sin(np.deg2rad(pas_inject[jj])) # mas
                        de_temp = seps_inject[ii]*pxsc*np.cos(np.deg2rad(pas_inject[jj])) # mas
                        dist = np.sqrt((ra-ra_temp)**2+(de-de_temp)**2) # mas
                        if (dist < mrad*pxsc):
                            flag = False
                            break
                    if (flag == True):
                        todo += [i*Npa+j]
                        done += [i*Npa+j]
                        stamp = np.array([offsetpsf*flux_inject[i] for k in range(dataset.input.shape[0])]) # MJy/sr
                        fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=stamp, astr_hdrs=dataset.wcs, radius=seps_inject[i], pa=pas_inject[j], field_dependent_correction=partial(utils.field_dependent_correction, meta=meta))
                        flux_all += [flux_inject[i]] # MJy/sr
                        seps_all += [seps_inject[i]] # pix
                        pas_all += [pas_inject[j]] # deg
        
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
                                  maxnumbasis=meta.maxnumbasis[key],
                                  psf_library=dataset.psflib,
                                  highpass=False,
                                  verbose=False)
        
        # Recover fake companions by fitting a 2D Gaussian.
        klipdataset = odir_temp+'FAKE_%04.0f_' % ctr+key+'-KLmodes-all.fits'
        with pyfits.open(klipdataset) as hdul:
            outputfile = hdul[0].data[0]
            outputfile_centers = [hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']]
        for k in range(len(todo)):
            jj = todo[k] % Npa
            ii = (todo[k]-jj)//Npa
            fake_flux = fakes.retrieve_planet_flux(frames=outputfile, centers=outputfile_centers, astr_hdrs=dataset.output_wcs[0], sep=seps_inject[ii], pa=pas_inject[jj], searchrad=5)
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
    y[x < p[1]] = 0.
    
    return y

def func_lnprob(p,x,y):
    return np.abs(y[:-1]-func(p, x[:-1]))
