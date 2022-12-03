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
from scipy.ndimage import gaussian_filter
from scipy import ndimage

import pyklip.instruments.JWST as JWST
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm
import pyklip.fitpsf as fitpsf
import pyklip.fakes as fakes
import pyklip.parallelized as parallelized
from pyklip.klip import nan_gaussian_filter
from copy import deepcopy

import webbpsf, webbpsf_ext
webbpsf_ext.setup_logging(level='ERROR', verbose=False)


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

    if (meta.ref_obs is not None) and isinstance(meta.ref_obs, (list,np.ndarray)):
        sci_ref_dir = 'SCI+REF'
    else:
        sci_ref_dir = 'SCI'

    # If necessary, extract the metadata of the observations.
    if (not meta.done_subtraction):
        if meta.comp_usefile == 'bgsub':
            subdir = 'IMGPROCESS/BGSUB'
        elif meta.use_cleaned:
            subdir = f'IMGPROCESS/{sci_ref_dir}_CLEAN'
        else:
            subdir = f'IMGPROCESS/{sci_ref_dir}'
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

            if filt in ['F250M']:
                #Don't use a fourier shift for these.
                fourier=False

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

            # Create a new pyKLIP dataset for forward modeling the companion
            # PSFs.
            if meta.repeatcentering_companion == False:
                centering_alg = 'savefile'
            else:
                centering_alg = meta.repeatcentering_companion

            if not hasattr(meta, 'blur_images'):
                meta.blur_images = False

            load_file0_center = meta.load_file0_center if hasattr(meta,'load_file0_center') else False

            if (meta.use_psfmask == True):
                try:
                    mask = fits.getdata(meta.psfmask[key], 'SCI') #NIRCam
                except:
                    try:
                        mask = fits.getdata(meta.psfmask[key], 0) #MIRI
                    except:
                        raise FileNotFoundError('Unable to read psfmask file {}'.format(meta.psfmask[key]))
            else:
                mask = None

            dataset = JWST.JWSTData(filepaths=filepaths, psflib_filepaths=psflib_filepaths, centering=meta.centering_alg,
                                    scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                    fiducial_point_override=meta.fiducial_point_override, blur=meta.blur_images,
                                    load_file0_center=load_file0_center,save_center_file=meta.ancildir+'shifts/file0_centers',
                                    spectral_type=meta.spt, mask=mask)
            # Get the coronagraphic mask transmission map.
            utils.get_transmission(meta, key, odir, derotate=False)

            # Get an offset PSF that is normalized to the total intensity of
            # the host star.
            if meta.offpsf == 'webbpsf':
                offsetpsf = utils.get_offsetpsf(meta, key,
                                                recenter_offsetpsf=recenter_offsetpsf,
                                                derotate=False, fourier=fourier)
                #kwd - just debugging
                if pxsc > 100:
                    inst = 'MIRI'
                    immask = 'FQPM{}'.format(filt[1:5])
                else:
                    inst = 'NIRCAM'
                    immask = key.split('_')[-1]
                if hasattr(meta, "psf_spec_file"):
                    if meta.psf_spec_file != False:
                        SED = io.read_spec_file(meta.psf_spec_file)
                    else:
                        SED = None
                offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                              sp=SED, use_coeff=True,
                                              date=meta.psfdate)
            elif meta.offpsf == 'webbpsf_ext':
                # Define some quantities
                if pxsc > 100:
                    inst = 'MIRI'
                    immask = 'FQPM{}'.format(filt[1:5])
                else:
                    inst = 'NIRCAM'
                    immask = key.split('_')[-1]
                if hasattr(meta, "psf_spec_file"):
                    if meta.psf_spec_file != False:
                        SED = io.read_spec_file(meta.psf_spec_file)
                    else:
                        SED = None
                else:
                    SED = None
                offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                              sp=SED, use_coeff=True,
                                              date=meta.psfdate)
                field_dep_corr = None #WebbPSF already corrects for transmissions.

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
                    guess_flux = 2.452e-4
                guess_spec = np.array([1.])

                if meta.offpsf == 'webbpsf_ext':
                    # Generate PSF for initial guess, if very different this could be garbage
                    # Negative sign on ra as webbpsf_ext expects in x,y space
                    offsetpsf = offsetpsf_func.gen_psf([-meta.ra_off[j]/1e3,meta.de_off[j]/1e3], do_shift=False, quick=False)

                    # TODO note that adding offsetpsf_func_input as an argument should be significantly faster,
                    # but currently does not work with parallelization for some reason (works if fm.py debug = True)
                    field_dep_corr = partial(utils.field_dependent_correction, meta=meta, offsetpsf_func_input=offsetpsf_func)

                if meta.offpsf == 'webbpsf':
                    # Generate PSF for initial guess, if very different this could be garbage
                    # Negative sign on ra as webbpsf_ext expects in x,y space
                    offsetpsf = offsetpsf_func.gen_psf([-meta.ra_off[j]/1e3,meta.de_off[j]/1e3], do_shift=False, quick=False)

                    field_dep_corr = partial(utils.field_dependent_correction, meta=meta)


                offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr

                if meta.blur_images != False:
                    offsetpsf = gaussian_filter(offsetpsf, meta.blur_images)

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
                                                field_dependent_correction=field_dep_corr)

                    # Compute the forward-modeled dataset.
                    annulus = meta.annuli #[[guess_sep-20., guess_sep+20.]] # pix
                    if len(annulus) == 1:
                        annulus = annulus[0]
                    try:
                        if annulus == 1:
                            annulus = [(0, dataset.input.shape[1]//2)]
                    except:
                        continue
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
                    stamp = ndimage.shift(offsetpsf, (shy, shx), mode='constant', cval=0.)

                    if meta.offpsf == 'webbpsf':
                        # Generate a position-dependent PSF from within field_dependent_correction (accounts for transmission)
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

                    # Write the images to a file for future plotting
                    io.save_fitpsf_images(odir+key+'-fitpsf.fits', fma)

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
                    if hasattr(meta, 'flux_percent_error'):
                        # Constant is from -2.5 / ln10
                        dpcentmag = 2.5/np.log(10)*(meta.flux_percent_error/100)
                        res[key][temp]['dappmag'] = np.sqrt(dstarmag**2+ddeltamag**2+dpcentmag**2)
                        res[key][temp]['dpcentmag'] = dpcentmag
                    else:
                        res[key][temp]['dappmag'] = np.sqrt(dstarmag**2+ddeltamag**2)
                    res[key][temp]['dstarmag'] = dstarmag
                    res[key][temp]['ddeltamag'] = ddeltamag

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

                    # check if pymultinest is installed:
                    # if not, print error message with conda link
                    # conda install -c conda-forge pymultinest
                    try:
                        fit = fitpsf.PlanetEvidence(guess_sep, guess_pa, fitboxsize, tempoutput)
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError('Pymultinest is not installed, try \n \"conda install -c conda-forge pymultinest\"')

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
                    evidence_ratio = fm_evidence - null_evidence

                    residnfig = fit.fm_residuals()
                    if (meta.plotting == True):
                        # savefile = odir+key+'-chains_c%.0f' % (j+1)+'.pdf'
                        # plotting.plot_chains(fma.sampler.chain, savefile)
                        corn, nullcorn = fit.fit_plots()
                        plt.close()
                        corn
                        plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
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
                    res[key][temp]['dappmag'] = np.sqrt((dstarmag)**2+(ddeltamag)**2)
                    res[key][temp]['evidence_ratio'] = evidence_ratio
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
                        print('   lnZ/Z0 = %.2f' % (res[key][temp]['evidence_ratio']))

        # Save the results
        compfile = odir+key+'-comp_save.json'
        with open(compfile, 'w') as sf:
            json.dump(res, sf)

    return res

def inject_fit(meta):
    '''
    Function to inject companions of a known flux into images and perform
    the PSF fitting. This can allow us to capture the true errors in the
    photometric measurements that might not be estimated correctl when using
    the diagonal kernel on data with correlated noise. extract_companions()
    must be run first

    Initial steps are very similar to extract companions,

    '''

    if (meta.ref_obs is not None) and isinstance(meta.ref_obs, (list,np.ndarray)):
        sci_ref_dir = 'SCI+REF'
    else:
        sci_ref_dir = 'SCI'

    # If necessary, extract the metadata of the observations we're injecting into
    if (not meta.done_subtraction):
        if meta.comp_usefile == 'bgsub':
            subdir = 'IMGPROCESS/BGSUB'
        elif meta.use_cleaned:
            subdir = f'IMGPROCESS/{sci_ref_dir}_CLEAN'
        else:
            subdir = f'IMGPROCESS/sci_ref_dir'
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

            if filt in ['F250M']:
                #Don't use a fourier shift for these.
                fourier=False

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

            # Create a new pyKLIP dataset for forward modeling the companion
            # PSFs.
            if meta.repeatcentering_companion == False:
                centering_alg = 'savefile'
            else:
                centering_alg = meta.repeatcentering_companion

            if not hasattr(meta, 'blur_images'):
                meta.blur_images = False

            load_file0_center = meta.load_file0_center if hasattr(meta,'load_file0_center') else False

            raw_dataset = JWST.JWSTData(filepaths=filepaths, psflib_filepaths=psflib_filepaths, centering=meta.centering_alg,
                                    scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                    fiducial_point_override=meta.fiducial_point_override, blur=meta.blur_images,
                                    load_file0_center=load_file0_center,save_center_file=meta.ancildir+'shifts/file0_centers',
                                    spectral_type=meta.spt, mask=mask)

            # Make the offset PSF
            if pxsc > 100:
                inst = 'MIRI'
                immask = 'FQPM{}'.format(filt[1:5])
            else:
                inst = 'NIRCAM'
                immask = key.split('_')[-1]
            if hasattr(meta, "psf_spec_file"):
                if meta.psf_spec_file != False:
                    SED = io.read_spec_file(meta.psf_spec_file)
                else:
                    SED = None
            else:
                SED = None
            offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                      sp=SED, use_coeff=True,
                                      date=meta.psfdate)

            field_dep_corr = None

            # Want to subtract the known companions using the fit from extract_companions()
            all_seps, all_flux = [], []
            with open(odir+key+'-comp_save.json', 'r') as f:
                compdata = json.load(f)[key]
                for comp in list(compdata.keys()):
                    # Get companion information
                    ra = compdata[comp]['ra']
                    de = compdata[comp]['de']
                    flux = compdata[comp]['f'] #Flux relative to star

                    sep = np.sqrt(ra**2+de**2) / pxsc
                    pa = np.rad2deg(np.arctan2(ra, de))

                    offsetpsf = offsetpsf_func.gen_psf([-ra/1e3,de/1e3], do_shift=False, quick=False)
                    if meta.blur_images != False:
                        offsetpsf = gaussian_filter(offsetpsf, meta.blur_images)
                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
                    offsetpsf *= flux #Negative flux as we are subtracting!

                    # Save some things
                    all_seps.append(sep)
                    all_flux.append(flux)

                    # Inject the planet as a negative source!
                    stamps = np.array([-offsetpsf for k in range(raw_dataset.input.shape[0])])
                    fakes.inject_planet(frames=raw_dataset.input, centers=raw_dataset.centers, inputflux=stamps, astr_hdrs=raw_dataset.wcs, radius=sep, pa=pa, field_dependent_correction=None)

            # Get some information from the original meta file in the run directory
            metasave = io.read_metajson(rdir+'SUBTRACTED/MetaSave.json')
            mode = metasave['used_mode']
            annuli = metasave['used_annuli']
            subsections = metasave['used_subsections']


            # Inject new planets at the same separations across many datasets
            # Right now only coding things up for 1 planet per image
            ctr = 0 #Keep count of injections
            for sep in all_seps:
                flux = all_flux[all_seps.index(sep)]
                # Loop over 8 different PAs
                pas = [195,209,223,237,251,266,280,294,309,323,337,351,5,19,33,49,62,76,91,105]
                for pa in pas:
                    # make a copy of the dataset above
                    dataset = deepcopy(raw_dataset)

                    # Now run KLIP to get the subtracted image
                    # Sometimes faster to just use 1 thread.
                    odir_temp = odir+'INJECTED/'
                    if (not os.path.exists(odir_temp)):
                        os.makedirs(odir_temp)

                    ra = sep*pxsc*np.sin(np.deg2rad(pa)) # mas
                    de = sep*pxsc*np.cos(np.deg2rad(pa)) # mas

                    # Need to make a new offsetpsf but can use old function
                    offsetpsf = offsetpsf_func.gen_psf([-ra/1e3,de/1e3], do_shift=False, quick=False)
                    if meta.blur_images != False:
                        offsetpsf = gaussian_filter(offsetpsf, meta.blur_images)
                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr

                    # Inject new source at desired PA
                    stamps = np.array([offsetpsf*flux for k in range(dataset.input.shape[0])])
                    fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=stamps, astr_hdrs=dataset.wcs, radius=sep, pa=pa, field_dependent_correction=None)

                    # KLIP the dataset
                    parallelized.klip_dataset(dataset=dataset,
                                              mode=mode,
                                              outputdir=odir_temp,
                                              fileprefix='FAKE_%04.0f_' % ctr+key,
                                              annuli=annuli,
                                              subsections=subsections,
                                              movement=1,
                                              numbasis=[meta.truenumbasis[key][KLindex]],
                                              calibrate_flux=False,
                                              numthreads=1,
                                              maxnumbasis=meta.maxnumbasis[key],
                                              psf_library=dataset.psflib,
                                              highpass=False,
                                              verbose=False)

                    # Loop through all companions.
                    res[key] = {}
                    for j in range(len(meta.ra_off)):

                        # Guesses for the fit parameters.
                        guess_sep = sep # pix
                        guess_pa = pa # deg
                        try:
                            guess_flux = meta.contrast_guess
                        except:
                            guess_flux = 2.452e-4
                        guess_spec = np.array([1.])

                        # Compute the forward-modeled dataset if it does not exist,
                        # yet, or if overwrite is True.
                        fmdataset = odir_temp+'FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
                        klipdataset = odir_temp+'FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
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
                                                         field_dependent_correction=field_dep_corr)

                            # Compute the forward-modeled dataset.
                            annulus = meta.annuli #[[guess_sep-20., guess_sep+20.]] # pix
                            if len(annulus) == 1:
                                annulus = annulus[0]
                            try:
                                if annulus == 1:
                                    annulus = [(0, dataset.input.shape[1]//2)]
                            except:
                                continue
                            subsection = 1
                            fm.klip_dataset(dataset=dataset,
                                            fm_class=fm_class,
                                            mode=mode,
                                            outputdir=odir_temp,
                                            fileprefix='FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key,
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

                            if (meta.plotting == True):
                                savefile = odir_temp+key+'-fmpsf_{}'.format(pa)+'c%.0f' % (j+1)+'.pdf'
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
                                fma.fit_astrometry(nwalkers=meta.nwalkers, nburn=meta.nburn, nsteps=meta.nsteps, numthreads=meta.numthreads, chain_output=odir_temp+key+'-bka_chain_{}'.format(pa)+'c%.0f' % (j+1)+'.pkl')
                                fma.sampler.chain[:, :, 0] *= pxsc
                                fma.sampler.chain[:, :, 1] *= pxsc
                                fma.sampler.chain[:, :, 2] *= guess_flux
                                if (meta.plotting == True):
                                    savefile = odir_temp+key+'-chains_{}'.format(pa)+'c%.0f' % (j+1)+'.pdf'
                                    plotting.plot_chains(fma.sampler.chain, savefile)
                                    fma.make_corner_plot()
                                    plt.savefig(odir_temp+key+'-corner_{}'.format(pa)+'%.0f' % (j+1)+'.pdf')
                                    plt.close()
                                    fma.best_fit_and_residuals()
                                    plt.savefig(odir_temp+key+'-model_{}'.format(pa)+'c%.0f' % (j+1)+'.pdf')
                                    plt.close()

                                # Write the images to a file for future plotting
                                io.save_fitpsf_images(odir_temp+key+'_{}'.format(pa)+'-fitpsf.fits', fma)

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

                            # Save the results
                            compfile = odir_temp+key+'_{}'.format(pa)+'-comp_save.json'
                            with open(compfile, 'w') as sf:
                                json.dump(res, sf)

                ctr+=1



    return

def planet_killer(meta, small_planet=None, recenter_offsetpsf=False,
                 use_fm_psf=True, fourier=True):
    '''
    Is your planet buried in the PSF of another planet, or co-incident source?
    Did you already fit a PSF to the brighter object using extract_companions?
    Use this function!

    Function to inject negative copies of fit companions and perform
    the PSF fitting on residual images.

    Initial steps are very similar to extract companions, inject_fit

    '''

    if (meta.ref_obs is not None) and isinstance(meta.ref_obs, (list,np.ndarray)):
        sci_ref_dir = 'SCI+REF'
    else:
        sci_ref_dir = 'SCI'

    # If necessary, extract the metadata of the observations we're injecting into
    if (not meta.done_subtraction):
        if meta.comp_usefile == 'bgsub':
            subdir = 'IMGPROCESS/BGSUB'
        elif meta.use_cleaned:
            subdir = f'IMGPROCESS/{sci_ref_dir}_CLEAN'
        else:
            subdir = f'IMGPROCESS/sci_ref_dir'
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

            if filt in ['F250M']:
                #Don't use a fourier shift for these.
                fourier=False

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

            # Create a new pyKLIP dataset for forward modeling the companion
            # PSFs.
            if meta.repeatcentering_companion == False:
                centering_alg = 'savefile'
            else:
                centering_alg = meta.repeatcentering_companion

            if not hasattr(meta, 'blur_images'):
                meta.blur_images = False

            load_file0_center = meta.load_file0_center if hasattr(meta,'load_file0_center') else False

            if (meta.use_psfmask == True):
                try:
                    mask = fits.getdata(meta.psfmask[key], 'SCI') #NIRCam
                except:
                    try:
                        mask = fits.getdata(meta.psfmask[key], 0) #MIRI
                    except:
                        raise FileNotFoundError('Unable to read psfmask file {}'.format(meta.psfmask[key]))
            else:
                mask = None

            raw_dataset = JWST.JWSTData(filepaths=filepaths, psflib_filepaths=psflib_filepaths, centering=meta.centering_alg,
                                    scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                    fiducial_point_override=meta.fiducial_point_override, blur=meta.blur_images,
                                    load_file0_center=load_file0_center,save_center_file=meta.ancildir+'shifts/file0_centers',
                                    spectral_type=meta.spt, mask=mask)

            # Make the offset PSF
            if pxsc > 100:
                inst = 'MIRI'
                immask = 'FQPM{}'.format(filt[1:5])
            else:
                inst = 'NIRCAM'
                immask = key.split('_')[-1]
            if hasattr(meta, "psf_spec_file"):
                if meta.psf_spec_file != False:
                    SED = io.read_spec_file(meta.psf_spec_file)
                else:
                    SED = None
            else:
                SED = None
            offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                      sp=SED, use_coeff=True,
                                      date=meta.psfdate)

            field_dep_corr = None

            # Want to subtract the known companions using the fit from extract_companions()
            all_seps, all_flux = [], []
            with open(odir+key+'-comp_save.json', 'r') as f:
                compdata = json.load(f)[key]
                for comp in list(compdata.keys()):
                    # Get companion information
                    ra = compdata[comp]['ra']
                    de = compdata[comp]['de']
                    flux = compdata[comp]['f'] #Flux relative to star

                    sep = np.sqrt(ra**2+de**2) / pxsc
                    pa = np.rad2deg(np.arctan2(ra, de))

                    offsetpsf = offsetpsf_func.gen_psf([-ra/1e3,de/1e3], do_shift=False, quick=False)
                    if meta.blur_images != False:
                        offsetpsf = gaussian_filter(offsetpsf, meta.blur_images)
                    offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
                    offsetpsf *= flux #Negative flux as we are subtracting!

                    # Save some things
                    all_seps.append(sep)
                    all_flux.append(flux)

                    # Inject the planet as a negative source!
                    stamps = np.array([-offsetpsf for k in range(raw_dataset.input.shape[0])])
                    fakes.inject_planet(frames=raw_dataset.input, centers=raw_dataset.centers, inputflux=stamps, astr_hdrs=raw_dataset.wcs, radius=sep, pa=pa, field_dependent_correction=None)

            # Get some information from the original meta file in the run directory
            metasave = io.read_metajson(rdir+'SUBTRACTED/MetaSave.json')
            mode = metasave['used_mode']
            annuli = metasave['used_annuli']
            subsections = metasave['used_subsections']


            # Reduce the companion subtracted images
            if small_planet is not None:
                sep = small_planet['sep']
                pa = small_planet['pa']
                flux = small_planet['flux']
                print('using input source position')
            else:
                sep = 3
                pa = 120
                flux = 0
            ctr = 0
            # make a copy of the dataset above
            dataset = deepcopy(raw_dataset)

            # Now run KLIP to get the subtracted image
            # Sometimes faster to just use 1 thread.
            odir_temp = odir+'INJECTED/'
            if (not os.path.exists(odir_temp)):
                os.makedirs(odir_temp)

            ra = sep*pxsc*np.sin(np.deg2rad(pa)) # mas
            de = sep*pxsc*np.cos(np.deg2rad(pa)) # mas

            # Need to make a new offsetpsf but can use old function
            offsetpsf = offsetpsf_func.gen_psf([-ra/1e3,de/1e3], do_shift=False, quick=False)
            if meta.blur_images != False:
                offsetpsf = gaussian_filter(offsetpsf, meta.blur_images)
            offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr

            # Inject new source at desired PA
            # if small_planet is not None:
            #     stamps = np.array([offsetpsf*0 for k in range(dataset.input.shape[0])])
            #     fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=stamps, astr_hdrs=dataset.wcs, radius=sep, pa=pa, field_dependent_correction=None)
            # else:
            #     stamps = np.array([offsetpsf*flux for k in range(dataset.input.shape[0])])
            #     fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=stamps, astr_hdrs=dataset.wcs, radius=sep, pa=pa, field_dependent_correction=None)

            # KLIP the dataset
            parallelized.klip_dataset(dataset=dataset,
                                      mode=mode,
                                      outputdir=odir_temp,
                                      fileprefix='KILLED_%04.0f_' % ctr+key,
                                      annuli=annuli,
                                      subsections=subsections,
                                      movement=1,
                                      numbasis=[meta.truenumbasis[key][KLindex]],
                                      calibrate_flux=False,
                                      numthreads=1,
                                      maxnumbasis=meta.maxnumbasis[key],
                                      psf_library=dataset.psflib,
                                      highpass=False,
                                      verbose=False)

            # Loop through all companions.
            res[key] = {}
            # the zeroth companion ... 
            j = -1

            # Guesses for the fit parameters.
            guess_sep = sep # pix
            guess_pa = pa # deg
            try:
                guess_flux = flux
            except:
                guess_flux = 2.452e-4
            guess_spec = np.array([1.])

            # Compute the forward-modeled dataset if it does not exist,
            # yet, or if overwrite is True.
            fmdataset = odir_temp+'FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
            klipdataset = odir_temp+'FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
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
                                             field_dependent_correction=field_dep_corr)

                # Compute the forward-modeled dataset.
                annulus = meta.annuli #[[guess_sep-20., guess_sep+20.]] # pix
                if len(annulus) == 1:
                    annulus = annulus[0]
                try:
                    if annulus == 1:
                        annulus = [(0, dataset.input.shape[1]//2)]
                except:
                    continue
                subsection = 1
                fm.klip_dataset(dataset=dataset,
                                fm_class=fm_class,
                                mode=mode,
                                outputdir=odir_temp,
                                fileprefix='FM_{}'.format(pa)+'-c%.0f-' % (j+1)+key,
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
                    stamp = ndimage.shift(offsetpsf, (shy, shx), mode='constant', cval=0.)

                    if meta.offpsf == 'webbpsf':
                        # Generate a position-dependent PSF from within field_dependent_correction (accounts for transmission)
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

                if (meta.nested == True):

                    # the fortran code here requires a shorter output file than is nominally passed
                    # so try shortening it and then moving the files after
                    tempoutput = './temp-multinest/'

                    # check if pymultinest is installed:
                    # if not, print error message with conda link
                    # conda install -c conda-forge pymultinest
                    try:
                        fit = fitpsf.PlanetEvidence(guess_sep, guess_pa, fitboxsize, tempoutput)
                    except ModuleNotFoundError:
                        raise ModuleNotFoundError('Pymultinest is not installed, try \n \"conda install -c conda-forge pymultinest\"')

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
                    evidence_ratio = fm_evidence - null_evidence

                    residnfig, c_j_dydx = fit.fm_residuals(return_residuals=True)
                    if (meta.plotting == True):
                        # savefile = odir+key+'-chains_c%.0f' % (j+1)+'.pdf'
                        # plotting.plot_chains(fma.sampler.chain, savefile)
                        corn, nullcorn = fit.fit_plots()
                        plt.close()
                        corn
                        plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
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
                    res[key][temp]['dappmag'] = np.sqrt((dstarmag)**2+(ddeltamag)**2)
                    res[key][temp]['evidence_ratio'] = evidence_ratio
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
                        print('   lnZ/Z0 = %.2f' % (res[key][temp]['evidence_ratio']))

    return
