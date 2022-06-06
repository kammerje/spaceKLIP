import os, re, sys

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

# Hard-coded parameters.
masks_nircam = ['MASKA210R', 'MASKA335R', 'MASKA430R', 'MASKALWB', 'MASKASWB']
masks_nircam_rnd = ['MASKA210R', 'MASKA335R', 'MASKA430R']
masks_nircam_bar = ['MASKALWB', 'MASKASWB']
masks_miri = ['LYOT', 'LYOT_2300', '4QPM', '4QPM_1065', '4QPM_1140', '4QPM_1550']

def extract_companions(meta):
    """
    Extract astrometry and photometry from any detected companions using
    the pyKLIP forward modeling class.
    
    TODO: tries to convert MJy/sr to contrast using the host star
          magnitude, but the results are wrong by a factor of ~5.
    
    TODO: until RDI is implemented for forward modeling, simply use the
          offset PSF with the correct coronagraphic mask transmission
          applied.
    
    TODO: use a position dependent offset PSF from pyNRC instead of the
          completely unocculted offset PSF from WebbPSF.

    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from
        engine.py.
    """

    if meta.verbose:
        print('--> Extracting companion properties...')

    # If necessary, build the obs dictionary etc
    if not meta.done_subtraction:
        basefiles = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS', search=meta.sub_ext)
        meta = utils.prepare_meta(meta, basefiles)
        # Set the subtraction flag for other stages
        meta.done_subtraction = True

    # Loop through all modes, numbers of annuli, and numbers of
    # subsections
    meta.truenumbasis = {}

    # Loop through directories of subtracted images
    for counter, rdir in enumerate(meta.rundirs):
        # Check if run directory actually exists
        if not os.path.exists(rdir):
            raise ValueError('Could not find provided run directory "{}"'.format(rdir))

        # Get the mode from the saved meta file
        metasave = io.read_metajson(rdir+'SUBTRACTED/MetaSave.json')
        mode = metasave['used_mode']

        # Define the input and output directories for each set of pyKLIP
        # parameters
        idir = rdir + 'SUBTRACTED/'
        odir = rdir + 'COMPANION/'
        if (not os.path.exists(odir)):
            os.makedirs(odir)

        # Create an output directory for the forward modeled datasets
        odir_temp = odir+'FITS/'
        if (not os.path.exists(odir_temp)):
            os.makedirs(odir_temp)

        # Loop through all sets of observing parameters
        res = {}
        for i, key in enumerate(meta.obs.keys()):
            meta.truenumbasis[key] = [num for num in meta.numbasis if (num <= meta.maxnumbasis[key])]

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
            hdul.close()

            # Create a new pyKLIP dataset for forward modeling the companion
            # PSFs
            dataset = JWST.JWSTData(filepaths=filepaths,
                                    psflib_filepaths=psflib_filepaths)

            # 2D map of the total throughput, i.e., an integration time
            # weighted average of the coronmsk transmission over the rolls
            utils.get_transmission(meta, pxsc, filt, mask, subarr, odir, key)

            # Offset PSF from WebbPSF, i.e., an integration time weighted
            # average of the unocculted offset PSF over the rolls normalized
            # to a total intensity of 1
            offsetpsf = utils.get_offsetpsf(meta, inst, filt, mask, key, derotate=False)
            offsetpsf *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
            
            # # Get the correct instrument from WebbPSF.
            # if (mask in masks_nircam):
            #     inst = webbpsf.NIRCam()
            # elif (mask in masks_miri):
            #     inst = webbpsf.MIRI()
            # else:
            #     raise UserWarning('Unknown coronagraphic mask')
            
            # # 
            # inst.filter = filt
            # if (mask in masks_nircam_rnd):
            #     inst.pupil_mask = 'MASKRND'
            # elif (mask in masks_nircam_bar):
            #     if (mask == 'MASKALWB'):
            #         inst.pupil_mask = 'MASKLWB'
            #     elif (mask == 'MASKASWB'):
            #         inst.pupil_mask = 'MASKSWB'
            #     inst.options['bar_offset'] = meta.bar_offset[key] # arcsec
            # elif (masks_miri):
            #     pass
            # else:
            #     raise UserWarning('Unknown coronagraphic mask')
            
            # Loop through all companions
            res[key] = {}
            for j in range(len(meta.ra_off)):

                # Guesses for the fit parameters
                guess_dx = meta.ra_off[j]/pxsc # pix
                guess_dy = meta.de_off[j]/pxsc # pix
                guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
                guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg
                guess_flux = 1e-4
                guess_spec = np.array([1.])

                # If overwrite is false, check whether the forward modeled
                # datasets have been computed already
                fmdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
                klipdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
                if meta.overwrite or (not os.path.exists(fmdataset) or not os.path.exists(klipdataset)):

                    # Initialize the forward modeling pyKLIP class
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
                                                 field_dependent_correction=partial(utils.correct_transmission, meta=meta))

                    # Compute the forward modeled datasets
                    annulus = [[guess_sep-20., guess_sep+20.]] # pix
                    subsection = 1
                    fm.klip_dataset(dataset=dataset,
                                    fm_class=fm_class,
                                    mode=mode,
                                    outputdir=odir_temp,
                                    fileprefix='FM_C%.0f-' % (j+1)+key,
                                    annuli=annulus,
                                    subsections=subsection,
                                    movement=1,
                                    numbasis=meta.truenumbasis[key],
                                    maxnumbasis=meta.maxnumbasis[key],
                                    calibrate_flux=False,
                                    psf_library=dataset.psflib,
                                    highpass=False,
                                    mute_progression=True)

                # Open the forward modeled datasets
                with pyfits.open(fmdataset) as hdul:
                    fm_frame = hdul[0].data[meta.KL]
                    fm_centx = hdul[0].header['PSFCENTX']
                    fm_centy = hdul[0].header['PSFCENTY']
                with pyfits.open(klipdataset) as hdul:
                    data_frame = hdul[0].data[meta.KL]
                    data_centx = hdul[0].header["PSFCENTX"]
                    data_centy = hdul[0].header["PSFCENTY"]
                    
                # The following lines create a PSF stamp from the position-
                # dependent transmission function (i.e., ignoring the RDI/ADI
                # algorithm throughput). They are not used anymore since
                # pyKLIP was updated to support RDI FM.
                # if mode == 'RDI':
                #     fm_frame = np.zeros_like(fm_frame)
                #     if fm_centx % 1. != 0. or fm_centy % 1. != 0.:
                #         raise UserWarning('Requires forward modeled PSF with integer center')
                #     else:
                #         fm_centx = int(fm_centx)
                #         fm_centy = int(fm_centy)
                #     sx, sy = offsetpsf.shape
                #     if sx % 2 != 1 or sy % 2 != 1:
                #         raise UserWarning('Requires offset PSF with odd shape')
                #     temp = shift(offsetpsf.copy(), (guess_dy-int(guess_dy), -(guess_dx-int(guess_dx))))
                #     rx = np.arange(sx)-sx//2+guess_dx
                #     ry = np.arange(sy)-sy//2+guess_dy
                #     xx, yy = np.meshgrid(rx, ry)
                #     temp = utils.correct_transmission(temp, xx, yy, meta)
                #     fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = temp # scale forward modeled PSF similar as in pyKLIP

                #     # Uncomment the following to use the PSF that was injected
                #     # by pyNRC
                #     # test = []
                #     # totet = 0. # s
                #     # for ii in range(len(ww_sci)):
                #     #     inttm = self.obs[key]['NINTS'][ww_sci[ii]]*self.obs[key]['EFFINTTM'][ww_sci[ii]] # s
                #     #     test += [inttm*rotate(np.load('../HIP65426/pynrc_figs/seq_000_filt_'+filt+'_psfs_%+04.0fdeg.npy' % self.obs[key]['PA_V3'][ww_sci[ii]])[j], -self.obs[key]['PA_V3'][ww_sci[ii]], reshape=False, mode='constant', cval=0.)]
                #     #     totet += inttm # s
                #     # test = np.sum(np.array(test), axis=0)/totet
                #     # test *= self.F0[filt]/10.**(mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
                #     # fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = test # scale forward modeled PSF similar as in pyKLIP
                # else:
                #     raise UserWarning('Not implemented yet')

                if meta.plotting:
                    savefile = odir+key+'-fmpsf_c%.0f' % (j+1)+'.pdf'
                    plotting.plot_fm_psf(meta, fm_frame, guess_flux, data_frame, pxsc=pxsc, j=j, savefile=savefile)

                # Fit the forward modeled PSF to the
                # KLIP-subtracted data.
                fitboxsize = 17 # pix
                fma = fitpsf.FMAstrometry(guess_sep=guess_sep,
                                          guess_pa=guess_pa,
                                          fitboxsize=fitboxsize)
                fma.generate_fm_stamp(fm_image=fm_frame,
                                      fm_center=[fm_centx, fm_centy],
                                      padding=5)
                fma.generate_data_stamp(data=data_frame,
                                        data_center=[data_centx, data_centy],
                                        dr=4,
                                        exclusion_radius=12.*fwhm)

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
                
                # Run the MCMC 
                fma.fit_astrometry(nwalkers=meta.nwalkers, nburn=meta.nburn, nsteps=meta.nsteps, numthreads=meta.numthreads, chain_output=odir+key+'-bka_chain_c%.0f' % (j+1)+'.pkl')
                
                if meta.plotting:
                    # Plot chains
                    savefile = odir+key+'-chains_c%.0f' % (j+1)+'.pdf'
                    plotting.plot_chains(fma.sampler.chain, savefile)

                    # Plot corner plot
                    fma.make_corner_plot()
                    plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
                    plt.close()
                    
                    # Plot best fit model psf and residuals
                    fma.best_fit_and_residuals()
                    plt.savefig(odir+key+'-model_c%.0f' % (j+1)+'.pdf')
                    plt.close()
                
                # Write best fit values into results dictionary.
                temp = 'c%.0f' % (j+1)
                res[key][temp] = {}
                res[key][temp]['ra'] = fma.raw_RA_offset.bestfit*pxsc # mas
                res[key][temp]['dra'] = fma.raw_RA_offset.error*pxsc # mas
                res[key][temp]['de'] = fma.raw_Dec_offset.bestfit*pxsc # mas
                res[key][temp]['dde'] = fma.raw_Dec_offset.error*pxsc # mas
                res[key][temp]['f'] = fma.raw_flux.bestfit
                res[key][temp]['df'] = fma.raw_flux.error
                
                if (meta.verbose == True):
                    print('Companion %.0f' % (j+1))
                    print('   RA  = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['ra'], res[key][temp]['dra'], meta.ra_off[j]))
                    print('   DEC = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['de'], res[key][temp]['dde'], meta.de_off[j]))
                    try:
                        condir = meta.idir[:meta.idir.find('/')]+'/pynrc_figs/'
                        confiles = [f for f in os.listdir(condir) if filt in f and f.endswith('_cons.npy')]
                        if (len(confiles) != 1):
                            raise UserWarning()
                        else:
                            con = np.load(condir+confiles[0])[j]
                        print('   CON = %.2e+/-%.2e (%.2e inj.)' % (res[key][temp]['f'], res[key][temp]['df'], con))
                    except:
                        print('   CON = %.2e+/-%.2e' % (res[key][temp]['f'], res[key][temp]['df']))
    
    return res
