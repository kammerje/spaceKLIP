from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import os
import importlib

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import astropy.units as u

from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import fourier_shift, rotate, shift
from scipy.optimize import minimize
from synphot import Observation, SourceSpectrum, SpectralElement
from synphot.models import Empirical1D
from synphot.units import convert_flux

import pyklip.instruments.JWST as JWST
import webbpsf, webbpsf_ext
from webbpsf_ext import robust

from jwst import datamodels
from jwst.datamodels import dqflags
from jwst.coron import AlignRefsStep

from . import io
from . import psf

rad2mas = 180./np.pi*3600.*1000.


# =============================================================================
# MAIN
# =============================================================================

def fourier_imshift(image, shift, pad=False, cval=0.0):
    """
    Fourier image shift. Adapted from JWST stage 3 pipeline.

    Parameters
    ----------
    image : array
        A 2D/3D image to be shifted.
    shift : array
        xshift, yshift.
    pad : bool
        Should we pad the array before shifting, then truncate?
        Otherwise, the image is wrapped.
    cval : sequence or float, optional
        The values to set the padded values for each axis. Default is 0.
        ((before_1, after_1), ... (before_N, after_N)) unique pad constants for each axis.
        ((before, after),) yields same before and after constants for each axis.
        (constant,) or int is a shortcut for before = after = constant for all axes.

    Returns
    -------
    offset : array
        Shifted image.

    """

    if (image.ndim == 2):

        ny, nx = image.shape

        # Pad border with zeros
        if pad:
            xshift, yshift = shift
            padx = np.abs(int(xshift)) + 5
            pady = np.abs(int(yshift)) + 5
            pad_vals = ([pady]*2,[padx]*2)
            image = np.pad(image, pad_vals, 'constant', constant_values=cval)
        else:
            padx = pady = 0

        shift = np.asanyarray(shift)[:2]
        offset_image = fourier_shift(np.fft.fftn(image), shift[::-1])
        offset = np.fft.ifftn(offset_image).real

        # Remove padded border to return to original size
        offset = offset[pady:pady+ny, padx:padx+nx]

    elif (image.ndim == 3):
        nslices = image.shape[0]
        shift = np.asanyarray(shift)[:, :2]
        if (shift.shape[0] != nslices):
            raise ValueError('The number of provided shifts must be equal to the number of slices in the input image')

        offset = np.empty_like(image, dtype=float)
        for k in range(nslices):
            offset[k] = fourier_imshift(image[k], shift[k], pad=pad, cval=cval)

    else:
        raise ValueError(f'Input image must be either a 2D or a 3D array. Found {image.ndim} dimensions.')

    return offset

def shift_invpeak(shift, image):
    """
    Shift an image and compute the inverse of its peak count.

    Parameters
    ----------
    shift : array
        xshift, yshift.
    image : array
        A 2D image to be shifted.

    Returns
    -------
    invpeak : float
        Inverse of the peak count of the shifted image.

    """

    # Fourier shift the image.
    offset = fourier_imshift(image, shift)

    # Compute the inverse of its peak count.
    invpeak = 1./np.max(offset)

    return invpeak

def recenter(image):
    """
    Recenter an image by shifting it around and minimizing the inverse of its
    peak count (i.e., maximizing its peak count).

    Parameters
    ----------
    image : array
        A 2D image to be recentered.

    Returns
    -------
    shift : array
        xshift, yshift.

    """

    # Find the shift that recenters the image.
    p0 = np.array([0., 0.])
    shift = minimize(shift_invpeak,
                     p0,
                     args=(image))['x']

    return shift

def get_offsetpsf(meta, key, recenter_offsetpsf=False, derotate=True,
                  fourier=True):
    """
    Get a derotated and integration time weighted average of an offset PSF
    from WebbPSF. Try to load it from the offsetpsfdir and generate it if it
    is not in there, yet. The offset PSF will be normalized to a total
    intensity of 1.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    key : str
        Dictionary key of the meta.obs dictionary specifying the considered
        concatenation.
    recenter_offsetpsf : bool
        Recenter the offset PSF? The offset PSF from WebbPSF is not properly
        centered because the wedge mirror that folds the light onto the
        coronagraphic subarrays introduces a chromatic shift.
    derotate : bool
        Derotate (and integreation time weigh) the offset PSF?
    fourier : bool
        Whether to perform shifts in the Fourier plane. This better preserves
        the total flux, however it can introduce Gibbs artefacts for the
        shortest NIRCAM filters as the PSF is undersampled.

    Returns
    -------
    totpsf : array
        Derotated and integration time weighted average of the offset PSF.

    """

    # Try to load the offset PSF from the offsetpsfdir and generate it if it
    # is not in there, yet.
    offsetpsfdir = meta.offsetpsfdir
    inst = meta.instrume[key]
    filt = meta.filter[key]
    mask = meta.coronmsk[key]

    try:
        with pyfits.open(offsetpsfdir+filt+'_'+mask+'.fits') as op_hdu:
            offsetpsf = op_hdu[0].data
    except:
        offsetpsf = gen_offsetpsf(meta, offsetpsfdir, inst, filt, mask)

    # Recenter the offset PSF.
    if (recenter_offsetpsf == True):
        shifts = recenter(offsetpsf)
        if fourier:
            offsetpsf = fourier_imshift(offsetpsf, shifts)
        else:
            offsetpsf = shift(offsetpsf, shifts, mode='constant', cval=0.)

    # Find the science target observations.
    ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]

    # Derotate the offset PSF and coadd it weighted by the integration time of
    # the different rolls.
    if (derotate == True):
        totpsf = np.zeros_like(offsetpsf)
        totexp = 0. # s
        for i in range(len(ww_sci)):
            totint = meta.obs[key]['NINTS'][ww_sci[i]]*meta.obs[key]['EFFINTTM'][ww_sci[i]] # s
            totpsf += totint*rotate(offsetpsf.copy(), meta.obs[key]['ROLL_REF'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
            totexp += totint # s
        totpsf /= totexp
    else:
        totpsf = offsetpsf

    return totpsf

def gen_offsetpsf(meta, offsetpsfdir, inst, filt, mask, xyoff=None, source=None, date=None):
    """
    Generate an offset PSF using WebbPSF and save it in the offsetpsfdir. The
    offset PSF will be normalized to a total intensity of 1.

    Parameters
    ----------
    offsetpsfdir : str
        Directory where the offset PSF shall be saved to.
    inst : str
        JWST instrument.
    filt : str
        JWST filter.
    mask : str
        JWST coronagraphic mask.
    xyoff : tuple
        Offset in detector coordinates (arcsec) from mask center to generate position-dependent psf 
    source: synphot.spectrum.SourceSpectrum 
        Default to 5700K blackbody if source=None
    date : str or None
        Date time in UTC as ISO-format string, a la 2022-07-01T07:20:00.
        If not set, then default webbpsf OPD is used (e.g., RevAA).

    """

    # NIRCam.
    if (inst == 'NIRCAM'):
        nircam = webbpsf.NIRCam()

        # Apply the correct pupil mask, but no image mask (unocculted PSF).
        if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
            nircam.pupil_mask = 'MASKRND'
        elif (mask in ['MASKALWB']):
            nircam.pupil_mask = 'MASKLWB'
        elif (mask in ['MASKASWB']):
            nircam.pupil_mask = 'MASKSWB'
        else:
            raise UserWarning('Unknown coronagraphic mask')
        nircam.image_mask = None
        webbpsf_inst = nircam
    # MIRI.
    elif (inst == 'MIRI'):
        miri = webbpsf.MIRI()
        if ('4QPM' in mask):
            miri.pupil_mask = 'MASKFQPM' #F not 4 for WebbPSF
        else:
            miri.pupil_mask = 'MASKLYOT'
        if xyoff: # assume if an offset is applied that the PSF should be relative to 4QPM
            miri.image_mask = mask.replace('4QPM_', 'FQPM')
        else:
            miri.image_mask = None #mask.replace('4QPM_', 'FQPM')
        webbpsf_inst = miri
    else:
        raise UserWarning('Unknown instrument')

    if hasattr(meta, "psf_spec_file"):
        if meta.psf_spec_file != False:
            SED = io.read_spec_file(meta.psf_spec_file)
        else:
            SED = None
    else:
        SED = None

    # Assign the correct filter:
    webbpsf_inst.filter = filt

    # If the PSF is position-dependent, offset from center:
    if xyoff:
        webbpsf_inst.options['source_offset_x'] = xyoff[0]
        webbpsf_inst.options['source_offset_y'] = xyoff[1]

    # Load date-specific OPD files? 
    date = None # test
    if date is not None: 
        print("----")
        print(f"loading time-dependent OPD of {date}! yay")
        print("----")
        webbpsf_inst.load_wss_opd_by_date(date=date, choice='before', verbose=False, plot=False)

    # Compute the offset PSF:
    hdul = webbpsf_inst.calc_psf(oversample=1, fov_pixels = 65, normalize='last', source=SED) # kwd - hardcode fov pix?

    # Save the offset PSF.
    if (not os.path.exists(offsetpsfdir)):
        os.makedirs(offsetpsfdir)
    # don't save it again if you don't have to:        
    if (not os.path.exists(offsetpsfdir+filt+'_'+mask+'.fits')):
        hdul.writeto(offsetpsfdir+filt+'_'+mask+'.fits')
    hdul.close()

    # Get the PSF array to return
    psf = hdul[0].data

    return psf

def get_transmission(meta, key, odir, derotate=False):
    """
    Get a derotated and integration time weighted average of a PSF mask and
    write it to meta.transmission.

    Note: assumes that the center of the PSF mask is aligned with the position
          of the host star PSF (except for the NIRCam bar masks).

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    key : str
        Dictionary key of the meta.obs dictionary specifying the considered
        concatenation.
    odir : str
        Directory where the PSF mask plot shall be saved to.
    derotate : bool
        Derotate (and integreation time weigh) the PSF mask?

    Returns
    -------
    totmsk : array
        Derotated and integration time weighted average of the PSF mask.

    """

    # Find the science target observations.
    ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]

    # Open the correct PSF mask. The assumption is that the center of the PSF
    # mask is aligned with the position of the host star PSF (except for the
    # NIRCam bar masks).
    psfmask = meta.psfmask[key]
    hdul = pyfits.open(psfmask)
    inst = meta.instrume[key]
    mask = meta.coronmsk[key]
    pxsc = meta.pixscale[key] # mas
   
    # NIRCam.
    if (inst == 'NIRCAM'):
        tp = hdul['SCI'].data[1:-1, 1:-1] # crop artifact at the edge

    # MIRI.
    elif (inst == 'MIRI'):
        if mask.lower() == '4qpm_1065':
            filt = 'f1065c'
        elif mask.lower() == '4qpm_1140':
            filt = 'f1140c'
        elif mask.lower() == '4qpm_1550':
            filt = 'f1550c'
        elif mask.lower() == 'lyot_2300':
            filt = 'f2300c'

        try:
            tp, _ = JWST.trim_miri_data([hdul['SCI'].data[None, :, :]], filt)
        except:
            tp, _ = JWST.trim_miri_data([hdul[0].data[None, :, :]], filt)
        tp = tp[0][0] #Just want the 2D array
        #tp = tp[0, 1:-1, 1:-2]

    else:
        raise UserWarning('Unknown instrument')
    hdul.close()

    # For the NIRCam bar masks, shift the PSF masks to their correct center.
    # Values outside of the subarray are filled with zeros (i.e., no
    # transmission).
    if (mask in ['MASKALWB', 'MASKASWB']):
        tp = shift(tp, (0., -meta.bar_offset[key]*1000./pxsc), mode='constant', cval=0.)
 
    # Derotate the PSF mask and coadd it weighted by the integration time of
    # the different rolls.
    if (derotate == True):
        totmsk = np.zeros_like(tp)
        totexp = 0. # s
        for i in range(len(ww_sci)):
            totint = meta.obs[key]['NINTS'][ww_sci[i]]*meta.obs[key]['EFFINTTM'][ww_sci[i]] # s
            totmsk += totint*rotate(tp.copy(), -meta.obs[key]['ROLL_REF'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
            totexp += totint # s
        totmsk /= totexp
    else:
        totmsk = tp

    # Create a regular grid interpolator taking 2D pixel offset as an input
    # and returning the coronagraphic mask transmission.
    xr = np.arange(tp.shape[1]) # pix
    yr = np.arange(tp.shape[0]) # pix

    xx, yy = np.meshgrid(xr, yr) # pix
    xx = xx-(tp.shape[1]-1.)/2. # pix
    yy = yy-(tp.shape[0]-1.)/2. # pix
    rr = np.sqrt(xx**2+yy**2) # pix
    totmsk[rr > meta.owa] = np.nan

    meta.transmission = RegularGridInterpolator((yy[:, 0], xx[0, :]), totmsk)
    if '335R' in mask:
        true_cen = [149.9-0.5, 174.4-0.5] #x, y, 0.5 offset
        shape = [320, 320]
        offset = [s/2-c for c,s in zip(true_cen, shape)]
        meta.current_mask_center_offset = offset
    elif '1140' in mask:
        true_cen = [119.749-13-0.5, 112.236-7-0.5] #x, y, offsets from pyKLIP
        shape = np.flip(tp.shape)
        offset = [s/2-c for c,s in zip(true_cen, shape)]
        meta.current_mask_center_offset = offset
    elif '1550' in mask:
        true_cen = [119.746-13-0.5, 113.289-8-0.5] #x, y, 0.5 offsets from pyKLIP
        shape = tp.shape
        offset = [s/2-c for c,s in zip(true_cen, shape)]
        meta.current_mask_center_offset = offset
    else:
        print('WARNING: Center offsets have not yet been determined for this mask!')
        meta.current_mask_center_offset = [0, 0]

    # Plot.
    plt.figure(figsize=(6.4, 4.8))
    ax = plt.gca()
    pp = ax.imshow(totmsk, origin='lower',
                   extent=(tp.shape[1]/2., -tp.shape[1]/2., -tp.shape[0]/2.,
                           tp.shape[0]/2.), cmap='viridis', vmin=0, vmax=1)
    cc = plt.colorbar(pp, ax=ax)
    cc.set_label('Transmission', rotation=270, labelpad=20)
    if (derotate == True):
        ax.set_xlabel('$\Delta$RA [pix]')
        ax.set_ylabel('$\Delta$Dec [pix]')
    else:
        ax.set_xlabel('$\Delta$x [pix]')
        ax.set_ylabel('$\Delta$y [pix]')
    ax.set_title('Transmission')
    plt.tight_layout()
    plt.savefig(odir+key+'-transmission.pdf')
    plt.close()

    return totmsk

def field_dependent_correction(stamp,
                               stamp_dx,
                               stamp_dy,
                               meta,
                               minx=0,
                               miny=0,
                               offsetpsf_func_input=None):
    """
    Apply the coronagraphic mask transmission to a PSF stamp.
    TEST REVISION (2022-8-12, KWD): 
        Make a *new* PSF stamp within the function as to generate an appropriate 
        position-dependent PSF using WebbPSF (or webbpsf_ext) directly.
        Note that this assumes that the input offset is provided relative 
        to the mask position, which is often NOT the star center!

    Note: assumes that the pyKLIP PSF center is the center of the
          coronagraphic mask transmission map.

    Note: uses a standard cartesian coordinate system so that North is +y and
          East is -x.

    Note: uses the coronagraphic mask transmission map stored in
          meta.transmission. Need to run get_transmission first!

    Parameters
    ----------
    stamp : array
        PSF stamp to which the coronagraphic mask transmission shall be
        applied. [NOT USED, since new PSF generated within this function]
    stamp_dx : array
        Array of the same shape as the PSF stamp containing the x-axis
        separation from the host star PSF center for each pixel.
    stamp_dy : array
        Array of the same shape as the PSF stamp containing the y-axis
        separation from the host star PSF center for each pixel.
    offset_func_input : function
        placeholder for webbpsf_ext function to pass, instead of generating
        multiple times in forward modeling

    Returns
    -------
    stamp : array
        PSF stamp to which the coronagraphic mask transmission was applied.

    """
    # generate a new stamp using regular webbpsf! 
    # TODO need to fix for instance with many keys (companions)!
    if len(meta.obs.keys()) == 1:
        key = list(meta.obs.keys())[0]
        inst = meta.instrume[key] 
        filt = meta.filter[key]
        mask = meta.coronmsk[key]
        pxar = meta.pixar_sr[key]

    offsetpsfdir = meta.offsetpsfdir

    # fix date for getting closest OPD - maybe this should not be forced?
    date = meta.psfdate

    # get appropriate pixel scale and mask nomenclature for instrument in question

    # NIRCam.
    if (inst == 'NIRCAM'):
        nircam = webbpsf.NIRCam()
        webbpsf_inst = nircam
        immask = key.split('_')[-1]
        pxscale = 0.063
    # MIRI.
    elif (inst == 'MIRI'):
        miri = webbpsf.MIRI()
        if ('4QPM' in mask):
            miri.pupil_mask = 'MASKFQPM' #F not 4 for WebbPSF
        else:
            miri.pupil_mask = 'MASKLYOT'
        # assume offset is applied that the PSF should be relative to 4QPM
        immask = mask.replace('4QPM_', 'FQPM')
        webbpsf_inst = miri 
        pxscale = 0.1108
    else:
        raise UserWarning('Unknown instrument')

    # Get center of input placeholder stamp
    c0 = (stamp.shape[0]-1)/2
    c1 = (stamp.shape[1]-1)/2 

    ostamp = stamp.copy()
    # generate stamp of psf at appropriate position relative to mask
    # use input offset from within pyklip fmpsf generate_models
    xyoff = (stamp_dx[int(c0),int(c1)]*pxscale - minx*pxscale, \
            stamp_dy[int(c0),int(c1)]*pxscale - miny*pxscale) # convert to arcsec for webbpsf
    print(f'Injected offset PSF at {xyoff} using the pixel scale of {pxscale} for {inst}')

    if meta.offpsf == 'webbpsf':

        # generate stamp with appropriate position-dep PSF (replaces input argument stamp)
        stamp = gen_offsetpsf(meta, offsetpsfdir, inst, filt, mask, xyoff=xyoff, date=date)

        # shift stamp image so that correct position-dep PSF is centered relative to stamp dimensions
        stamp = fourier_imshift(stamp, (-1*xyoff[0]/pxscale, -1*xyoff[1]/pxscale)) # convert back into px for shift

        # I believe we need to scale here as well to get the flux right
        stamp *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
        #print("the maximum value of the stamp after fourier_imshift and scaling is: ", stamp.max())

    elif meta.offpsf == 'webbpsf_ext':
        if hasattr(meta, "psf_spec_file"):
            if meta.psf_spec_file != False:
                SED = io.read_spec_file(meta.psf_spec_file)
            else:
                SED = None
        else:
            SED = None

        # see if function already generated upstream (faster)
        if offsetpsf_func_input is None:
            offsetpsf_func = psf.JWST_PSF(inst, filt, immask, fov_pix=65,
                                              sp=SED, use_coeff=True,
                                              date=meta.psfdate)
        else:
            offsetpsf_func = offsetpsf_func_input

        # use webbpsf_ext to generate psf in idl coordinates relative to aperture center
        # note this is automatically centered                                     
        stamp = offsetpsf_func.gen_psf_idl([xyoff[0], xyoff[1]], do_shift=False, quick=False)

        # normalize stamp to sum to 1.0, a la regular webbpsf
        stamp /= np.sum(stamp)

        # convert to flux
        stamp *= meta.F0[filt]/10.**(meta.mstar[filt]/2.5)/1e6/pxar # MJy/sr
    
    # As the model PSF's are centered on a pixel, need to
    # shift by the subpixel offset for this particular sep/pa
    # as provided by pyKLIP
    stamp = fourier_imshift(stamp, (-minx, -miny))

    # Also need to smooth the PSF if blur images was specified
    if meta.blur_images != False:
        stamp = gaussian_filter(stamp, meta.blur_images)


    # print(stamp.shape)
    # print(ostamp.shape)
    # fig, axs = plt.subplots(nrows=1, ncols=3)
    # axs[0].imshow(ostamp, vmin=np.nanmin(stamp-ostamp), vmax=np.nanmax(ostamp))
    # axs[0].set_title('Original')
    # axs[1].imshow(stamp, vmin=np.nanmin(stamp-ostamp), vmax=np.nanmax(ostamp))
    # axs[1].set_title('Replaced')
    # axs[2].set_title('Difference')
    # axs[2].imshow(stamp-ostamp, vmin=np.nanmin(stamp-ostamp), vmax=np.nanmax(ostamp))
    # plt.show()
    # exit()
    #stamp = ostamp

    # Note the following is deprecated by virtue of using the correct position-dependent PSF.
    # -----
    # Apply coronagraphic mask transmission.
    # need to check this relative to the input coordinates!
    # xy = np.vstack((stamp_dy.flatten(), stamp_dx.flatten())).T
    # transmission = meta.transmission(xy)
    # transmission = transmission.reshape(stamp.shape)

    # # Get center of stamp
    # c0 = (stamp.shape[0]-1)/2
    # c1 = (stamp.shape[1]-1)/2

    # Get transmission at this point
    #transmission_at_center = transmission[int(c0),int(c1)]

    ## Old way use peak of flux
    # peak_index = np.unravel_index(stamp.argmax(), stamp.shape)
    # transmission_at_center =  transmission[peak_index[1],peak_index[0]]
    # -----
    
    print('Generated a new psf from within field_dependent_correction')
    return stamp
    

def get_stellar_magnitudes(meta):
    # First find out if a file was provided correctly
    if not os.path.isfile(meta.sdir):
        # Not a valid input
        raise ValueError('Stellar directory not recognised, please supply a valid filepath.')

    # Get the instrument
    instrument = list(meta.obs.keys())[0].split('_')[0] #Just use first filter, NIRCam / MIRI run separately

    # Check if this is a Vizier VOTable, if so, use webbpsf_ext
    if meta.sdir[-4:] == '.vot':
        # Pick an arbitrary bandpass+magnitude to normalise the initial SED. Exact choice
        # doesn't matter as will be be refitting this model using the provided data.
        # Important thing is the spectral type (user provided).
        bp_k = webbpsf_ext.bp_2mass('k')
        bp_mag = 5

        # Magnitude value is arbitrary, as we will be using the Vizier photometry to renormalise and fit the SED.
        spec = webbpsf_ext.spectra.source_spectrum(name='Input Data + SED', sptype=meta.spt, mag_val=bp_mag, bp=bp_k, votable_file=meta.sdir)

        # Want to adjust where we fit the spectrum based on the observing filter, just roughly split between NIRCam and MIRI
        if instrument == 'NIRCAM' :
            wlim = [1,5]
        elif instrument == 'MIRI':
            wlim = [10, 20]

        # Fit the SED to the selected data
        spec.fit_SED(x0=[1.0], wlim=wlim, use_err=False, verbose=False) #Don't use the error as it breaks thing, and don't print scaling value.
        # spec.plot_SED()
        # plt.show()

        # Want to convert the flux to photlam so that it matches the per photon throughputs?
        input_flux = u.Quantity(spec.sp_model.flux, str(spec.sp_model.fluxunits))
        photlam_flux = convert_flux(spec.sp_model.wave, input_flux, out_flux_unit='photlam')

        # Spectrum is originally from pysynphot (outdated), convert to synphot.
        SED = SourceSpectrum(Empirical1D, points=spec.sp_model.wave << u.Unit(str(spec.sp_model.waveunits)), lookup_table=photlam_flux << u.Unit('photlam'))
    # If not a VOTable, try to read it in.
    else:
        SED = io.read_spec_file(meta.sdir)

    ### Now, perform synthetic observations on the SED to get stellar magnitudes
    # Get the filters used from the input datasets
    filters = [i.split('_')[2] for i in list(meta.obs.keys())]
    if instrument == 'NIRCAM':
        if ('F335M' not in filters):
            filters += ['F335M'] # make sure that TA filter is present

    # Calculate magnitude in each filter
    mstar = {}
    for filt in filters:
        # Read in the bandpass correctly
        with importlib.resources.open_text(f'spaceKLIP.resources.PCEs.{instrument}', f'{filt}.txt') as bandpass_file:
            bandpass_data = np.genfromtxt(bandpass_file).transpose()
            bandpass_wave = bandpass_data[0] * 1e4 #Convert from microns to angstrom
            bandpass_throughput = bandpass_data[1]

        # Create the bandpass object
        Bandpass = SpectralElement(Empirical1D, points=bandpass_wave, lookup_table=bandpass_throughput)

        # Perform synthetic observation
        Obs = Observation(SED, Bandpass, binset=Bandpass.waveset)
        VegaSED = SourceSpectrum.from_vega()
        magnitude = Obs.effstim(flux_unit='vegamag', vegaspec=VegaSED).value

        #magnitude = Obs.effstim(flux_unit='Jy', vegaspec=VegaSED).value

        # Add magnitude to dictionary
        mstar[filt.upper()] = magnitude

    # temporary feature until we figure out better file formatting with grant's models
    if hasattr(meta,'starmagerrs'):
        i = 0
        meta.dmstar = {}
        for filt in filters:
            meta.dmstar[filt] = meta.starmagerrs[i]
            i += 1

    return mstar

def get_maxnumbasis(meta):
    """
    Find the maximum numbasis based on the number of available calibrator
    frames.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    Returns
    -------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    """

    # Find the maximum numbasis based on the number of available calibrator
    # frames.
    meta.maxnumbasis = {}
    meta.adimaxnumbasis = {}
    meta.rdimaxnumbasis = {}
    for key in meta.obs.keys():
        # Get all science and calibrator keys
        ws = meta.obs[key]['TYP'] == 'SCI'
        wc = meta.obs[key]['TYP'] == 'CAL'
        # Sum integrations in science (except for 1 as we won't self reference)
        nbs = np.sum(meta.obs[key]['NINTS'][ws][1:], dtype=int) #
        # Sum all integrations in calibration (Reference)
        nbc = np.sum(meta.obs[key]['NINTS'][wc][:], dtype=int)
        meta.maxnumbasis[key] = nbs + nbc
        meta.adimaxnumbasis[key] = nbs
        meta.rdimaxnumbasis[key] = nbc

    return meta

def get_psfmasknames(meta):
    """
    Get the correct PSF mask for each concatenation using functionalities of
    the JWST pipeline.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    Returns
    -------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    """

    # Create an instance of the reference star alignment JWST pipeline step.
    # This just serves as a dummy from which the get_reference_file function
    # can be used to obtain any reference file type from the online CRDS
    # database.
    step = AlignRefsStep()

    # Get the correct PSF mask for each concatenation.
    meta.psfmask = {}

    _get_transmission_path = lambda filename: importlib.resources.path('spaceKLIP.resources.transmissions', filename)

    for key in meta.obs.keys():
        if '1065' in key:
            with _get_transmission_path('JWST_MIRI_F1065C_transmission_webbpsf-ext_v2.fits') as p:
                meta.psfmask[key] = str(p)
        elif '1140' in key:
            with _get_transmission_path('JWST_MIRI_F1140C_transmission_webbpsf-ext_v2.fits') as p:
                meta.psfmask[key] = str(p)
        elif '1550' in key:
            with _get_transmission_path('JWST_MIRI_F1550C_transmission_webbpsf-ext_v2.fits') as p:
                meta.psfmask[key] = str(p)
        elif 'MASK335R' in key:
            with _get_transmission_path('jwst_nircam_psfmask_mask335r_shift.fits') as p:
                meta.psfmask[key] = str(p)
        else:
            model = datamodels.open(meta.obs[key]['FITSFILE'][0])
            meta.psfmask[key] = step.get_reference_file(model, 'psfmask')
    del step

    return meta

def get_bar_offset(meta):
    """
    Get the correct bar offset for each concatenation from the meta object
    which contains the pySIAF bar offsets for the different NIRCam bar mask
    fiducial points in meta.offset_lwb and meta.offset_swb.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    Returns
    -------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    """

    # Get the correct bar offset for each concatenation.
    meta.bar_offset = {}
    for key in meta.obs.keys():
        if (meta.instrume[key] == 'NIRCAM'):
            if ('LWB' in meta.coronmsk[key]):
                if ('NARROW' in meta.apername[key]):
                    meta.bar_offset[key] = meta.offset_lwb['narrow']
                else:
                    meta.bar_offset[key] = meta.offset_lwb[meta.filter[key]]
            elif ('SWB' in meta.coronmsk[key]):
                if ('NARROW' in meta.apername[key]):
                    meta.bar_offset[key] = meta.offset_swb['narrow']
                else:
                    meta.bar_offset[key] = meta.offset_swb[meta.filter[key]]
            else: # round masks
                meta.bar_offset[key] = None
        else:
            meta.bar_offset[key] = None

    return meta

def prepare_meta(meta, fitsfiles):
    """
    Find and write the metadata for the provided FITS files into the meta
    object. This function overwrites any metadata that was previously stored
    in the meta object.

    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    fitsfiles : list of str
        List of the FITS files whose metadata shall be extracted.

    Returns
    -------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.

    """

    # Extract the metadata of the observations from the FITS files.
    meta = io.extract_obs(meta, fitsfiles)

    # Find the maximum numbasis based on the number of available calibrator
    # frames.
    meta = get_maxnumbasis(meta)

    # Find the names of the PSF masks from CRDS.
    meta = get_psfmasknames(meta)

    # Get the bar offsets for NIRCam from pySIAF.
    meta = get_bar_offset(meta)

    # Compute the host star magnitude in the observed filters.
    meta.mstar = get_stellar_magnitudes(meta)

    return meta


def bp_fix(im, sigclip=5, niter=1, pix_shift=1, rows=True, cols=True, 
           bpmask=None, return_mask=False, verbose=False, in_place=True):
    """ Find and fix bad pixels in image with median of surrounding values

    Slight modification of routine from pynrc.
    
    Paramters
    ---------
    im : ndarray
        Single image
    sigclip : int
        How many sigma from mean doe we fix?
    niter : int
        How many iterations for sigma clipping? 
        Ignored if bpmask is set.
    pix_shift : int
        We find bad pixels by comparing to neighbors and replacing.
        E.g., if set to 1, use immediate adjacents neighbors.
        Replaces with a median of surrounding pixels.
    rows : bool
        Compare to row pixels? Setting to False will ignore pixels
        along rows during comparison. Recommended to increase
        ``pix_shift`` parameter if using only rows or cols.
    cols : bool
        Compare to column pixels? Setting to False will ignore pixels
        along columns during comparison. Recommended to increase
        ``pix_shift`` parameter if using only rows or cols.
    bpmask : boolean array
        Use a pre-determined bad pixel mask for fixing.
    return_mask : bool
        If True, then also return a masked array of bad
        pixels where a value of 1 is "bad".
    verbose : bool
        Print number of fixed pixels per iteration
    in_place : bool
        Do in-place corrections of input array.
        Otherwise, return a copy.
    """

    def shift_array(im, pix_shift, rows=True, cols=True):
        '''Create an array of shifted values'''

        ny, nx = im.shape

        # Pad image
        padx = pady = pix_shift
        pad_vals = ([pady]*2,[padx]*2)
        im_pad = np.pad(im, pad_vals, mode='edge')

        shift_arr = []
        sh_vals = np.arange(pix_shift*2+1) - pix_shift
        # Set shifting of columns and rows
        xsh_vals = sh_vals if rows else [0]
        ysh_vals = sh_vals if cols else [0]
        for i in xsh_vals:
            for j in ysh_vals:
                if (i != 0) or (j != 0):
                    shift_arr.append(np.roll(im_pad, (j,i), axis=(0,1)))
        shift_arr = np.asarray(shift_arr)
        return shift_arr[:,pady:pady+ny,padx:padx+nx]
    
    # Only single iteration if bpmask is set
    if bpmask is not None:
        niter = 1
    
    if in_place:
        arr_out = im
    else:
        arr_out = im.copy()
    maskout = np.zeros(im.shape, dtype='bool')
    
    for ii in range(niter):
        # Create an array of shifted values
        shift_arr = shift_array(arr_out, pix_shift, rows=rows, cols=cols)
    
        # Take median of shifted values
        shift_med = np.nanmedian(shift_arr, axis=0)
        if bpmask is None:
            # Difference of median and reject outliers
            diff = arr_out - shift_med
            shift_std = robust.medabsdev(shift_arr, axis=0)

            indbad = diff > (sigclip*shift_std)
        else:
            indbad = bpmask

        # Mark anything that is a NaN
        indbad[np.isnan(arr_out)] = True
        
        # Set output array and mask values 
        arr_out[indbad] = shift_med[indbad]
        maskout[indbad] = True
        
        if verbose:
            print(f'Bad Pixels fixed: {indbad.sum()}')

        # No need to iterate if all pixels fixed
        if indbad.sum()==0:
            break
            
    if return_mask:
        return arr_out, maskout
    else:
        return arr_out

def clean_data(data, dq_masks, sigclip=5, niter=5, in_place=True, **kwargs):
    """Clean a data cube using bp_fix routine
    
    Iterative pixel fixing routine to clean bad pixels flagged in the DQ mask
    as well as spatial outliers. Replaces bad pixels with median of surrounding
    unflagged (good) pixels. Assumes anything with values of 0 were previously
    cleaned by the jwst pipeline and will be replaced with more representative 
    values.

    Parameters
    ----------
    data : ndarray
        Image cube.
    dq_masks : ndarray
        Data quaity array same size as data.
    sigclip : int
        How many sigma from mean doe we fix?
    niter : int
        How many iterations for sigma clipping? 
        Ignored if bpmask is set.
    in_place : bool
        Do in-place corrections of input array.
        Otherwise, works on a copy.

    Keyword Args
    ------------
    pix_shift : int
        Size of border pixels to compare to.
        We find bad pixels by comparing to neighbors and replacing.
        E.g., if set to 1, use immediate adjacents neighbors.
        Replaces with a median of surrounding pixels.
    rows : bool
        Compare to row pixels? Setting to False will ignore pixels
        along rows during comparison. Recommended to increase
        ``pix_shift`` parameter if using only rows or cols.
    cols : bool
        Compare to column pixels? Setting to False will ignore pixels
        along columns during comparison. Recommended to increase
        ``pix_shift`` parameter if using only rows or cols.
    verbose : bool
        Print number of fixed pixels per iteration
    """

    if not in_place:
        data = data.copy()

    sh = data.shape
    ndim_orig = len(sh)
    if ndim_orig==2:
        ny, nx = sh
        nz = 1
        data = data.reshape([nz,ny,nx])
    else:
        nz, ny, nx = sh
   
    for i in range(nz):

        im = data[i]
        # Get average background rate and standard deviation
        bg_med = np.nanmedian(im)
        bg_std = robust.medabsdev(im)
        # Clip bright PSFs for final bg calc
        bg_ind = im<(bg_med+10*bg_std)
        bg_med = np.nanmedian(im[bg_ind])
        bg_std = robust.medabsdev(im[bg_ind])

        # Create initial mask of 0s, NaNs, and large negative values
        bp_mask = (im==0) | np.isnan(im) | (im<bg_med-sigclip*bg_std)
        for k in ['DO_NOT_USE']:#, 'SATURATED', 'JUMP_DET']:
            mask = (dq_masks[i] & dqflags.pixel[k]) > 0
            bp_mask = bp_mask | mask

        # Flag out-of-family spatial outliers
        _, bp_mask2 = bp_fix(im, sigclip=sigclip, in_place=False, niter=niter, return_mask=True, **kwargs)
        bp_mask = bp_mask | bp_mask2

        # Fix only those pixels flagged in the input mask
        data[i] = bp_fix(im, bpmask=bp_mask, in_place=True, niter=10, **kwargs)
        # Final pass of additional median clipping
        data[i] = bp_fix(data[i], sigclip=sigclip, in_place=True, niter=niter, **kwargs)
        
    # Return back to 2-dimensional image if that was the input
    if ndim_orig==2:
        data = data.reshape([ny,nx])

    return data

def clean_file(file, sigclip=5, niter=5, **kwargs):
    """Clean the data in a file using bp_fix routine"""

    from astropy.io import fits
    hdul = fits.open(file)
    data = clean_data(hdul['SCI'].data, hdul['DQ'].data, sigclip=sigclip, niter=niter, **kwargs)
    hdul['SCI'].data = data
    hdul.writeto(file, overwrite=True)
    hdul.close()

