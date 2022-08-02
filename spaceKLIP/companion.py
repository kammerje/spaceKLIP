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
from scipy.integrate import simpson
from scipy.interpolate import interp1d
# from scipy.ndimage import shift
from scipy import ndimage

import pyklip.instruments.JWST as JWST
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
from pyklip.klip import nan_gaussian_filter

import webbpsf
webbpsf.setup_logging(level='ERROR')
import webbpsf_ext

from . import utils
from . import io
from . import plotting
from . import psf

# =============================================================================
# MAIN
# =============================================================================

def extract_companions(meta, recenter_offsetpsf=False, use_fm_psf=True,
                       fourier=True):
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
    fourier : bool
        [meta.offpsf = 'webbpsf] Whether to perform the recentering shift in
        the Fourier plane. This better preserves the total flux, however it can
        introduce Gibbs artefacts for the shortest NIRCAM filters as the PSF is
        undersampled.
    """

    if (meta.verbose == True):
        print('--> Extracting companion properties...')

    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        if meta.comp_usefile == 'bgsub':
            subdir = 'IMGPROCESS/BGSUB'
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
        odir = rdir+'COMPANION_KL{}/'.format(meta.KL)
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
            weff = meta.weff[filt] # m
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
            else:
                centering_alg = meta.repeatcentering_companion
            dataset = JWST.JWSTData(filepaths=filepaths,
                                    psflib_filepaths=psflib_filepaths, centering=centering_alg, badpix_threshold=meta.badpix_threshold,
                                    scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                    fiducial_point_override=meta.fiducial_point_override)

            # Get the coronagraphic mask transmission map.
            utils.get_transmission(meta, key, odir, derotate=False)

            # Get an offset PSF that is normalized to the total intensity of
            # the host star.
            if meta.offpsf == 'webbpsf':
                offsetpsf = utils.get_offsetpsf(meta, key,
                                                recenter_offsetpsf=recenter_offsetpsf,
                                                derotate=False, fourier=fourier)
                offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
            elif meta.offpsf == 'webbpsf_ext':
                # Define some quantities
                if pxsc > 100:
                    inst = 'MIRI'
                else:
                    inst = 'NIRCAM'
                if inst == 'MIRI':
                    immask = 'FQPM{}'.format(filt[1:5])
                offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                              sp=None, use_coeff=True,
                                              date=meta.psfdate)

            # Loop through all companions.
            res[key] = {}
            for j in range(len(meta.ra_off)):

                # Guesses for the fit parameters.
                guess_dx = meta.ra_off[j]/pxsc # pix
                guess_dy = meta.de_off[j]/pxsc # pix
                guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
                guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg
                try:
                    guess_flux = meta.contrast_guess
                except:
                    guess_flux = 1e-4
                guess_spec = np.array([1.])

                if meta.offpsf == 'webbpsf_ext':
                    # Generate PSF for initial guess, if very different this could be garbage
                    # Negative sign on ra as webbpsf_ext expects in x,y space
                    offsetpsf = offsetpsf_func.gen_psf([-meta.ra_off[j]/1e3,meta.de_off[j]/1e3], do_shift=False, quick=False)
                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr

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
                    annulus = meta.annuli#[[guess_sep-20., guess_sep+20.]] # pix
                    if len(annulus) == 1:
                        annulus = annulus[0]
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
                from pyklip.klip import nan_gaussian_filter
                with pyfits.open(fmdataset) as hdul:
                    fm_frame = hdul[0].data[KLindex]
                    try:
                        fm_frame = nan_gaussian_filter(fm_frame, meta.smooth)
                    except:
                        pass
                    fm_centx = hdul[0].header['PSFCENTX']
                    fm_centy = hdul[0].header['PSFCENTY']
                with pyfits.open(klipdataset) as hdul:
                    data_frame = hdul[0].data[KLindex]
                    try:
                        data_frame = nan_gaussian_filter(data_frame, meta.smooth)
                    except:
                        pass
                    data_centx = hdul[0].header["PSFCENTX"]
                    data_centy = hdul[0].header["PSFCENTY"]

                # If use_fm_psf is False, replace the forward-modeled PSF in
                # the fm_frame with a simple offset PSF from WebbPSF.

                if (use_fm_psf == False):
                    # Get a derotated and integration time weighted average of
                    # an offset PSF from WebbPSF. Apply the field-dependent
                    # correction and insert it at the correct companion
                    # position into the fm_frame.
                    if meta.offpsf == 'webbpsf_ext':
                        # Generate PSF for initial guess, if very different this could be garbage
                        # Negative sign on ra as webbpsf_ext expects in x,y space
                        offsetpsf = offsetpsf_func.gen_psf([-meta.ra_off[j]/1e3,meta.de_off[j]/1e3], do_shift=False, quick=False)
                    else:
                        # Get the coronagraphic mask transmission map.
                        utils.get_transmission(meta, key, odir, derotate=True)
                        offsetpsf = utils.get_offsetpsf(meta, key,
                                                        recenter_offsetpsf=recenter_offsetpsf,
                                                        derotate=True, fourier=fourier)

                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
                    offsetpsf *= guess_flux
                    sx = offsetpsf.shape[1]
                    sy = offsetpsf.shape[0]
                    if ((sx % 2 != 1) or (sy % 2 != 1)):
                        raise UserWarning('Offset PSF needs to be of odd shape')
                    shx = (fm_centx-int(fm_centx))-(guess_dx-int(guess_dx))
                    shy = (fm_centy-int(fm_centy))+(guess_dy-int(guess_dy))
                    stamp = shift(offsetpsf, (shy, shx), mode='constant', cval=0.)

                    if meta.offpsf == 'webbpsf':
                        # Need to multiply offaxis by coronagraph transmission
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
                # there's gotta be a better way to do this...
                try:
                    fitboxsize = meta.fitboxsize
                except:
                    fitboxsize = 25 # pix
                try:
                    dr = meta.dr
                except:
                    dr = 5
                try:
                    exc_rad = meta.exc_rad
                except:
                    exc_rad = 3
                try:
                    corr_len_guess = meta.corr_len_guess
                except:
                    corr_len_guess = 3.
                try:
                    x_range = meta.x_range
                except:
                    x_range = 2. # pix
                try:
                    y_range = meta.y_range
                except:
                    y_range = 2. # pix
                try:
                    flux_range = meta.flux_range
                except:
                    flux_range = 10. # mag
                try:
                    corr_len_range = meta.corr_len_range
                except:
                    corr_len_range = 1. # mag

                if (meta.mcmc == True):

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
                    corr_len_label = r'$l$'
                    try:
                        fma.set_kernel(meta.fitkernel, [corr_len_guess], [corr_len_label])
                    except:
                        fma.set_kernel("diag", [corr_len_guess], [corr_len_label])
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

                    deltamag = -2.5*np.log10(fma.fit_flux.bestfit*guess_flux)
                    ddeltamag = 2.5/np.log(10)*(fma.fit_flux.error*guess_flux)/(fma.fit_flux.bestfit*guess_flux)
                    starmag = meta.mstar[filt]
                    try:
                        dstarmag = meta.dmstar[filt]
                    except:
                        print('No stellar magnitude errors supplied yet so assuming +/- 0.1 mag (FIX TBD!)')
                        dstarmag = 0.1
                    res[key][temp]['appmag'] = starmag+deltamag
                    res[key][temp]['dappmag'] = np.sqrt((dstarmag/starmag)**2+(ddeltamag/deltamag)**2)*res[key][temp]['appmag']
 
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
                        print('   APPMAG = %.2e+/-%.2e' % (res[key][temp]['appmag'], res[key][temp]['dappmag']))


                if (meta.nested == True):

                    # the fortran code here requires a shorter output file than is nominally passed
                    # so try shortening it and then moving the files after
                    tempoutput = './temp-multinest/'
                    fit = fitpsf.PlanetEvidence(guess_sep, guess_pa, fitboxsize, tempoutput)
                    print('created PE module')

                    # generate FM stamp
                    fit.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

                    # generate data_stamp stamp
                    fit.generate_data_stamp(data_frame, [data_centx, data_centy], dr=dr, exclusion_radius=exc_rad*fwhm)
                    print('generated FM & data stamps')
                    # set kernel, no read noise
                    try:
                        corr_len_guess = meta.corr_len_guess
                    except:
                        corr_len_guess = 3.
                    corr_len_label = "l"
                    try:
                        fit.set_kernel(meta.fitkernel, [corr_len_guess], [corr_len_label])
                    except:
                        fit.set_kernel("diag", [corr_len_guess], [corr_len_label])
                    print('set kernel')

                    fit.set_bounds(x_range, y_range, flux_range, [corr_len_range])
                    print('set bounds')
                    #Run the pymultinest fit

                    fit.multifit()
                    print('ran fit')

                    evidence = fit.fit_stats()

                    #Forward model evidence
                    fm_evidence = evidence[0]['nested sampling global log-evidence']
                    #forward model parameter distributions, containing the median and percentiles for each
                    fm_posteriors = evidence[0]['marginals']
                    #Null model evidence
                    null_evidence = evidence[1]['nested sampling global log-evidence']
                    #null parameter distributions, containing the median and percentiles for each
                    null_posteriors = evidence[1]['marginals']
                    evidence_ratio = np.exp(fm_evidence - null_evidence)

                    print('evidence ratio is: ',round(np.log(evidence_ratio), 4),' >5 is strong evidence')
                    residnfig = fit.fm_residuals()
                    if (meta.plotting == True):
                        # savefile = odir+key+'-chains_c%.0f' % (j+1)+'.pdf'
                        # plotting.plot_chains(fma.sampler.chain, savefile)
                        corn, nullcorn = fit.fit_plots()
                        plt.close()
                        corn
                        plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
                        plt.close()
                        nullcorn
                        plt.savefig(odir+key+'-nullcorner_c%.0f' % (j+1)+'.pdf')
                        plt.close()
                        fit.fm_residuals()
                        plt.savefig(odir+key+'-model_c%.0f' % (j+1)+'.pdf')
                        plt.close()

                    # move multinest output from temp dir to odir+key
                    import shutil
                    tempfiles = os.listdir(tempoutput)
                    for f in tempfiles:
                        shutil.move(tempoutput + f, odir+key+f)
                    shutil.rmtree(tempoutput)

                    # Write the best fit values into the results dictionary.
                    temp = 'c%.0f' % (j+1)
                    res[key][temp] = {}
                    res[key][temp]['ra'] = -(fit.fit_x.bestfit-data_centx)*pxsc # mas
                    res[key][temp]['dra'] = (fit.fit_x.error)*pxsc # mas
                    res[key][temp]['de'] = (fit.fit_y.bestfit-data_centy)*pxsc # mas
                    res[key][temp]['dde'] = (fit.fit_y.error)*pxsc # mas
                    res[key][temp]['f'] = fit.fit_flux.bestfit*guess_flux
                    res[key][temp]['df'] = fit.fit_flux.error*guess_flux

                    deltamag = -2.5*np.log10(fit.fit_flux.bestfit*guess_flux)
                    ddeltamag = 2.5/np.log(10)*(fit.fit_flux.error*guess_flux)/(fit.fit_flux.bestfit*guess_flux)
                    starmag = meta.mstar[filt]
                    try:
                        dstarmag = meta.dmstar[filt]
                    except:
                        print('No stellar magnitude errors supplied yet so assuming +/- 0.1 mag (FIX TBD!)')
                        dstarmag = 0.1
                    res[key][temp]['appmag'] = starmag+deltamag
                    res[key][temp]['dappmag'] = np.sqrt((dstarmag/starmag)**2+(ddeltamag/deltamag)**2)*res[key][temp]['appmag']

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
                        print('   APPMAG = %.2f+/-%.2f' % (res[key][temp]['appmag'], res[key][temp]['dappmag']))

        # Save the results
        compfile = odir+key+'-comp_save.json'
        with open(compfile, 'w') as sf:
            json.dump(res, sf)

    return res
