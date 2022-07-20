from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================
import os, re, sys
import json

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from scipy.ndimage import shift

import pyklip.instruments.JWST as JWST
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf

import webbpsf
webbpsf.setup_logging(level='ERROR')

from . import utils
from . import io 
from . import plotting


# =============================================================================
# MAIN
# =============================================================================

def extract_companions(meta, recenter_offsetpsf=False, use_fm_psf=True):
    """
    Extract astrometry and photometry from any detected companions using
    the pyKLIP forward modeling class.
    
    TODO: use a position-dependent offset PSF from pyNRC instead of the
          completely unocculted offset PSF from WebbPSF.
    
    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    recenter_offsetpsf : bool
        Recenter the offset PSF? The offset PSF from WebbPSF is not properly
        centered because the wedge mirror that folds the light onto the
        coronagraphic subarrays introduces a chromatic shift.
    use_fm_psf : bool
        Use a pyKLIP forward-modeled offset PSF instead of a simple offset PSF.
    
    """
    
    if (meta.verbose == True):
        print('--> Extracting companion properties...')
    
    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        if meta.usebgsub_companion:
            subdir = 'IMGPROCESS+BGSUB'
        else:
            subdir = 'IMGPROCESS'
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir=subdir, search=meta.sub_ext)
        meta = utils.prepare_meta(meta, basefiles)
        meta.done_subtraction = True # set the subtraction flag for the subsequent pipeline stages
    
    # Loop through all directories of subtracted images.
    meta.truenumbasis = {}
    for counter, rdir in enumerate(meta.rundirs):
        # Check if run directory actually exists
        if not os.path.exists(rdir):
            raise ValueError('Could not find provided run directory "{}"'.format(rdir))

        # Get the mode from the saved meta file
        if (meta.verbose == True):
            dirparts = rdir.split('/')[-2].split('_') # -2 because of trailing '/'
            print('--> Mode = {}, annuli = {}, subsections = {}, scenario {} of {}'.format(dirparts[3], dirparts[4], dirparts[5], counter+1, len(meta.rundirs)))
        
        # Get the mode from the saved meta file.
        metasave = io.read_metajson(rdir+'SUBTRACTED/MetaSave.json')
        mode = metasave['used_mode']
        
        # Define the input and output directories for each set of pyKLIP
        # parameters.
        idir = rdir+'SUBTRACTED/'
        odir = rdir+'COMPANION/'
        if (not os.path.exists(odir)):
            os.makedirs(odir)
        
        # Create an output directory for the forward-modeled datasets.
        odir_temp = odir+'FITS/'
        if (not os.path.exists(odir_temp)):
            os.makedirs(odir_temp)

        # Loop through all concatenations.
        res = {}
        for i, key in enumerate(meta.obs.keys()):
            
            ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]
            filepaths = np.array(meta.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
            ww_cal = np.where(meta.obs[key]['TYP'] == 'CAL')[0]
            psflib_filepaths = np.array(meta.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
            hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
            filt = meta.filter[key]
            pxsc = meta.pixscale[key] # mas
            pxar = meta.pixar_sr[key] # sr
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
            # Get the index of the KL component we are interested in
            try:
                KLindex = all_numbasis.index(meta.KL)
            except:
                raise ValueError('KL={} not found. Calculated options are: {}, and maximum possible for this data is {}'.format(meta.KL, all_numbasis, meta.maxnumbasis[key]))
            meta.truenumbasis[key] = [num for num in all_numbasis if (num <= meta.maxnumbasis[key])]
            hdul.close()
            
            # Create a new pyKLIP dataset for forward modeling the companion
            # PSFs.
            if meta.repeatcentering_companion == False:
                centering_alg = 'savefile'
                print('Loading shifts from saved file...')
            else:
                centering_alg = meta.repeatcentering_companion
            dataset = JWST.JWSTData(filepaths=filepaths,
                                    psflib_filepaths=psflib_filepaths, centering=centering_alg, badpix_threshold=meta.badpix_threshold,
                                    scishiftfile=meta.ancildir+'scishifts', refshiftfile=meta.ancildir+'refshifts')
            
            # Get the coronagraphic mask transmission map.
            utils.get_transmission(meta, key, odir, derotate=False)
            
            # Get an offset PSF that is normalized to the total intensity of
            # the host star.
            offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=recenter_offsetpsf, derotate=False)
            offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr

            # plt.imshow(offsetpsf)
            # plt.show()

            # Loop through all companions.
            res[key] = {}
            for j in range(len(meta.ra_off)):
                # Guesses for the fit parameters.
                guess_dx = meta.ra_off[j]/pxsc # pix
                guess_dy = meta.de_off[j]/pxsc # pix
                guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
                guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg
                guess_flux = 1e-4
                guess_spec = np.array([1.])
                
                # Compute the forward-modeled dataset if it does not exist,
                # yet, or if overwrite is True.
                fmdataset = odir_temp+'FM_c%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
                klipdataset = odir_temp+'FM_c%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
                if ((meta.overwrite == True) or (not os.path.exists(fmdataset) or not os.path.exists(klipdataset))):
                    
                    # Initialize the forward modeling pyKLIP class.
                    input_wvs = np.unique(dataset.wvs)
                    if (len(input_wvs) != 1):
                        raise UserWarning('Only works with broadband photometry')
                    fm_class = fmpsf.FMPlanetPSF(inputs_shape=dataset.input.shape,
                                                 numbasis=np.array(meta.truenumbasis[key]),
                                                 sep=guess_sep,
                                                 pa=guess_pa,
                                                 dflux=guess_flux,
                                                 input_psfs=np.array([offsetpsf]),
                                                 input_wvs=input_wvs,
                                                 spectrallib=[guess_spec],
                                                 spectrallib_units='contrast',
                                                 field_dependent_correction=partial(utils.field_dependent_correction, meta=meta))
                    
                    # Compute the forward-modeled dataset.
                    annulus = [[guess_sep-20., guess_sep+20.]] # pix
                    subsection = 1
                    fm.klip_dataset(dataset=dataset,
                                    fm_class=fm_class,
                                    mode=mode,
                                    outputdir=odir_temp,
                                    fileprefix='FM_c%.0f-' % (j+1)+key,
                                    annuli=annulus,
                                    subsections=subsection,
                                    movement=1,
                                    numbasis=meta.truenumbasis[key],
                                    maxnumbasis=meta.maxnumbasis[key],
                                    calibrate_flux=False,
                                    psf_library=dataset.psflib,
                                    highpass=False,
                                    mute_progression=True)
                
                # Open the forward-modeled dataset.
                with pyfits.open(fmdataset) as hdul:
                    fm_frame = hdul[0].data[KLindex]
                    fm_centx = hdul[0].header['PSFCENTX']
                    fm_centy = hdul[0].header['PSFCENTY']
                with pyfits.open(klipdataset) as hdul:
                    data_frame = hdul[0].data[KLindex]
                    data_centx = hdul[0].header["PSFCENTX"]
                    data_centy = hdul[0].header["PSFCENTY"]
                
                # If use_fm_psf is False, replace the forward-modeled PSF in
                # the fm_frame with a simple offset PSF from WebbPSF.
                if (use_fm_psf == False):
                    
                    # Get the coronagraphic mask transmission map.
                    utils.get_transmission(meta, key, odir, derotate=True)
                    
                    # Get a derotated and integration time weighted average of
                    # an offset PSF from WebbPSF. Apply the field-dependent
                    # correction and insert it at the correct companion
                    # position into the fm_frame.
                    offsetpsf = utils.get_offsetpsf(meta, key, recenter_offsetpsf=recenter_offsetpsf, derotate=True)
                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
                    offsetpsf *= guess_flux
                    sx = offsetpsf.shape[1]
                    sy = offsetpsf.shape[0]
                    if ((sx % 2 != 1) or (sy % 2 != 1)):
                        raise UserWarning('Offset PSF needs to be of odd shape')
                    shx = (fm_centx-int(fm_centx))-(guess_dx-int(guess_dx))
                    shy = (fm_centy-int(fm_centy))+(guess_dy-int(guess_dy))
                    stamp = shift(offsetpsf, (shy, shx), mode='constant', cval=0.)
                    xx = np.arange(sx)-sx//2-(int(guess_dx)+(fm_centx-int(fm_centx)))
                    yy = np.arange(sy)-sy//2+(int(guess_dy)-(fm_centy-int(fm_centy)))
                    stamp_dx, stamp_dy = np.meshgrid(xx, yy)
                    stamp = utils.field_dependent_correction(stamp, stamp_dx, stamp_dy, meta)
                    fm_frame[:, :] = 0.
                    fm_frame[int(fm_centy)+int(guess_dy)-sy//2:int(fm_centy)+int(guess_dy)+sy//2+1, int(fm_centx)-int(guess_dx)-sx//2:int(fm_centx)-int(guess_dx)+sx//2+1] = stamp
                
                if (meta.plotting == True):
                    savefile = odir+key+'-fmpsf_c%.0f' % (j+1)+'.pdf'
                    plotting.plot_fm_psf(meta, fm_frame, data_frame, guess_flux, pxsc=pxsc, j=j, savefile=savefile)
                
                # Fit the forward-modeled PSF to the KLIP-subtracted data.
                fitboxsize = 25 # pix
                dr = 5
                exc_rad = 3
                fma = fitpsf.FMAstrometry(guess_sep=guess_sep,
                                          guess_pa=guess_pa,
                                          fitboxsize=fitboxsize)
                fma.generate_fm_stamp(fm_image=fm_frame,
                                      fm_center=[fm_centx, fm_centy],
                                      padding=5)
                fma.generate_data_stamp(data=data_frame,
                                        data_center=[data_centx, data_centy],
                                        dr=dr,
                                        exclusion_radius=exc_rad*fwhm)
                corr_len_guess = 3. # pix
                corr_len_label = r'$l$'
                fma.set_kernel('matern32', [corr_len_guess], [corr_len_label])
                x_range = 1. # pix
                y_range = 1. # pix
                flux_range = 1. # mag
                corr_len_range = 1. # mag
                fma.set_bounds(x_range, y_range, flux_range, [corr_len_range])
                
                # Make sure that noise map is invertible.
                noise_map_max = np.nanmax(fma.noise_map)
                fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
                fma.noise_map[fma.noise_map == 0.] = noise_map_max
                
                # Run the MCMC fit.
                fma.fit_astrometry(nwalkers=meta.nwalkers, nburn=meta.nburn, nsteps=meta.nsteps, numthreads=meta.numthreads, chain_output=odir+key+'-bka_chain_c%.0f' % (j+1)+'.pkl')
                fma.sampler.chain[:, :, 0] *= pxsc
                fma.sampler.chain[:, :, 1] *= pxsc
                fma.sampler.chain[:, :, 2] *= guess_flux
                if (meta.plotting == True):
                    savefile = odir+key+'-chains_c%.0f' % (j+1)+'.pdf'
                    plotting.plot_chains(fma.sampler.chain, savefile)
                    fma.make_corner_plot()
                    plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
                    plt.close()
                    fma.best_fit_and_residuals()
                    plt.savefig(odir+key+'-model_c%.0f' % (j+1)+'.pdf')
                    plt.close()
                
                # Write the best fit values into the results dictionary.
                temp = 'c%.0f' % (j+1)
                res[key][temp] = {}
                res[key][temp]['ra'] = fma.raw_RA_offset.bestfit*pxsc # mas
                res[key][temp]['dra'] = fma.raw_RA_offset.error*pxsc # mas
                res[key][temp]['de'] = fma.raw_Dec_offset.bestfit*pxsc # mas
                res[key][temp]['dde'] = fma.raw_Dec_offset.error*pxsc # mas
                res[key][temp]['f'] = fma.raw_flux.bestfit*guess_flux
                res[key][temp]['df'] = fma.raw_flux.error*guess_flux
                
                if (meta.verbose == True):
                    print('--> Companion %.0f' % (j+1))
                    print('   RA  = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['ra'], res[key][temp]['dra'], meta.ra_off[j]))
                    print('   DEC = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['de'], res[key][temp]['dde'], meta.de_off[j]))
                    try:
                        name = ['b', 'c', 'd', 'e']
                        pdir = meta.idir[:meta.idir[:-1].rfind('/')]+'/pmags/'
                        file = [f for f in os.listdir(pdir) if f.lower().endswith(name[j]+'_'+filt.lower()+'.npy')][0]
                        pmag = np.load(pdir+file)
                        dmag = pmag-meta.mstar[filt]
                        cinj = 10**(dmag/(-2.5))
                        print('   CON = %.2e+/-%.2e (%.2e inj.)' % (res[key][temp]['f'], res[key][temp]['df'], cinj))
                    except:
                        print('   CON = %.2e+/-%.2e' % (res[key][temp]['f'], res[key][temp]['df']))

        # Save the results
        compfile = odir+key+'-comp_save.json'
        with open(compfile, 'w') as sf:
            json.dump(res, sf)

    return res
