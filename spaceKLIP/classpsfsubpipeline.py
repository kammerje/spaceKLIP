from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pdb
import sys

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import gaussian_filter, rotate
from scipy.ndimage import shift as spline_shift
from scipy.optimize import leastsq
from spaceKLIP import utils as ut

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def subtractlsq(shift,
                image,
                ref_image,
                mask=None):
    
    res = image - shift[0] * ref_image
    res = res - gaussian_filter(res, 3)
    if mask is None:
        return res.ravel()
    else:
        return res[mask]

def run_obs(Database,
            kwargs={},
            subdir='psfsub'):
    
    # Set output directory.
    output_dir = os.path.join(Database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through concatenations.
    for i, key in enumerate(Database.obs.keys()):
        log.info('--> Concatenation ' + key)
        
        # Find science and reference files.
        ww_sci = np.where(Database.obs[key]['TYPE'] == 'SCI')[0]
        if len(ww_sci) == 0:
            raise UserWarning('Could not find any science files')
        ww_ref = np.where(Database.obs[key]['TYPE'] == 'REF')[0]
        if len(ww_ref) == 0:
            raise UserWarning('Could not find any reference files')
        
        # Loop through reference files.
        ref_data = []
        ref_erro = []
        ref_pxdq = []
        for j in ww_ref:
            
            # Read reference file.
            fitsfile = Database.obs[key]['FITSFILE'][j]
            data, erro, pxdq, head_pri, head_sci, is2d = ut.read_obs(fitsfile)
            
            # Compute median reference.
            ref_data += [data]
            ref_erro += [erro]
            ref_pxdq += [pxdq]
        ref_data = np.concatenate(ref_data)
        ref_erro = np.concatenate(ref_erro)
        ref_pxdq = np.concatenate(ref_pxdq)
        ref_data = np.nanmedian(ref_data, axis=0)
        Nsample = np.sum(np.logical_not(np.isnan(ref_erro)), axis=0)
        ref_erro = np.true_divide(np.sqrt(np.nansum(ref_erro**2, axis=0)), Nsample)
        if Database.obs[key]['TELESCOP'][j] == 'JWST' and Database.obs[key]['INSTRUME'][j] == 'NIRCAM':
            ref_pxdq = np.sum(ref_pxdq != 0, axis=0) != 0
        else:
            ref_pxdq = np.sum(ref_pxdq & 1 == 1, axis=0) != 0
        
        # Loop through science files.
        if i == 0:
            pps = []
        sci_data = []
        for j in ww_sci:
            
            # Read science file.
            fitsfile = Database.obs[key]['FITSFILE'][j]
            data, erro, pxdq, head_pri, head_sci, is2d = ut.read_obs(fitsfile)
            
            # Compute median science.
            data = np.nanmedian(data, axis=0)
            
            test = []
            for k in np.logspace(-2, 2, 100):
            # for k in np.linspace(0.03, 0.05, 100):
                temp = data - k * ref_data
                # temp = temp - gaussian_filter(temp, 3)
                test += [temp]
            test = np.array(test)
            hdu0 = pyfits.PrimaryHDU(test)
            hdul = pyfits.HDUList([hdu0])
            hdul.writeto(os.path.join(output_dir, key + '_test.fits'), output_verify='fix', overwrite=True)
            hdul.close()
            
            if i == 0:
                mask = np.zeros_like(data)
                mask[60:90, 20:60] = 1
                mask[140:170, 185:225] = 1
                mask = mask > 0.5
                p0 = np.array([1.])
                pp = leastsq(subtractlsq,
                              p0,
                              args=(data, ref_data, mask))[0]
                pps += [pp]
            else:
                pp = np.mean(pps)
            # pp = np.linspace(0.03, 0.05, 100)[79]
            print(pp)
            
            # if i == 0:
            #     psfmask = pyfits.getdata('/Users/jkammerer/Documents/Code/spaceKLIP/spaceKLIP/resources/transmissions/JWST_MIRI_F1550C_transmission_webbpsf-ext_v2.fits')
            # else:
            #     psfmask = pyfits.getdata('/Users/jkammerer/Documents/Code/spaceKLIP/spaceKLIP/resources/transmissions/jwst_miri_psfmask_0009.fits')
            #     psfmask[:, 144 - 5:144 + 6] = 0.
            temp = data - pp * ref_data
            shift = (temp.shape[1]//2 - Database.obs[key]['CRPIX1'][j] + 1 + Database.obs[key]['XOFFSET'][j] / Database.obs[key]['PIXSCALE'][j], temp.shape[0]//2 - Database.obs[key]['CRPIX2'][j] + 1 + Database.obs[key]['YOFFSET'][j] / Database.obs[key]['PIXSCALE'][j])
            temp = spline_shift(temp, shift[::-1])
            # psfmask = spline_shift(psfmask, shift[::-1])
            temp = rotate(temp, -Database.obs[key]['ROLL_REF'][j], reshape=False)
            # if i == 0:
            #     psfmask = rotate(psfmask, -Database.obs[key]['ROLL_REF'][j], reshape=False)
            # else:
            #     psfmask = rotate(psfmask, -Database.obs[key]['ROLL_REF'][j] + 4.7, reshape=False)
            # temp[psfmask < 0.6] = np.nan
            sci_data += [temp]
        sci_data = np.array(sci_data)
        sci_data = np.nanmedian(sci_data, axis=0)
        
        # Write FITS file.
        hdu0 = pyfits.PrimaryHDU(sci_data)
        hdul = pyfits.HDUList([hdu0])
        hdul.writeto(os.path.join(output_dir, key + '_psfsub.fits'), output_verify='fix', overwrite=True)
        hdul.close()
    
    pdb.set_trace()
    
    pass
