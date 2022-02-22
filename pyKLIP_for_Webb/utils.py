import os

import numpy as np
from scipy.ndimage import rotate

import webbpsf

def get_offsetpsf(offsetpsfdir,
                  obs,
                  filt,
                  mask,
                  key):
    """
    Get a derotated and integration time weighted average of an offset PSF
    from WebbPSF. Try to load it from the offsetpsfdir and generate it if
    it is not in there yet. This offset PSF is normalized to an integrated
    source of 1 and takes into account the throughput of the pupil mask.
    
    Parameters
    ----------
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    key: str
        Dictionary key of the self.obs dictionary specifying the
        considered observation.
    
    Returns
    -------
    totop: array
        Stamp of the derotated and integration time weighted average of
        the offset PSF.
    """
    
    # Try to load the offset PSF from the offsetpsfdir and generate it if
    # it is not in there yet.
    if (not os.path.exists(offsetpsfdir)):
        os.makedirs(offsetpsfdir)
    if (not os.path.exists(offsetpsfdir+filt+'_'+mask+'.npy')):
        gen_offsetpsf(offsetpsfdir, filt, mask)
    offsetpsf = np.load(offsetpsfdir+filt+'_'+mask+'.npy')
    
    # Find the science target observations.
    ww_sci = np.where(obs[key]['TYP'] == 'SCI')[0]
    
    # Compute the derotated and integration time weighted average of the
    # offset PSF. Values outside of the PSF stamp are filled with zeros.
    totop = np.zeros_like(offsetpsf)
    totet = 0. # s
    for i in range(len(ww_sci)):
        inttm = obs[key]['NINTS'][ww_sci[i]]*obs[key]['EFFINTTM'][ww_sci[i]] # s
        totop += inttm*rotate(offsetpsf.copy(), -obs[key]['PA_V3'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
        totet += inttm # s
    totop /= totet
    
    return totop

def gen_offsetpsf(offsetpsfdir,
                  filt,
                  mask):
    """
    Generate an offset PSF using WebbPSF. This offset PSF is normalized to
    an integrated source of 1 and takes into account the throughput of the
    pupil mask.
    
    Parameters
    ----------
    filt: str
        Filter name from JWST data header.
    mask: str
        Coronagraphic mask name from JWST data header.
    """
    
    # Get NIRCam
    nircam = webbpsf.NIRCam()

    # Apply the correct pupil mask, but no image mask (unocculted PSF).
    nircam.filter = filt
    if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
        nircam.pupil_mask = 'MASKRND'
    elif (mask in ['MASKASWB']):
        nircam.pupil_mask = 'MASKSWB'
    elif (mask in ['MASKALWB']):
        nircam.pupil_mask = 'MASKLWB'
    else:
        raise UserWarning()
    nircam.image_mask = None
    
    # Compute the offset PSF using WebbPSF and save it to the offsetpsfdir.
    hdul = nircam.calc_psf(oversample=1)
    psf = hdul[0].data # PSF center is at (39, 39)

    hdul.close()
    np.save(offsetpsfdir+filt+'_'+mask+'.npy', psf)
    
    return None