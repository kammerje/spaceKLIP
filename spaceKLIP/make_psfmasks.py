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

import json

from scipy.ndimage import shift

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

# Set parameters.
path = 'resources/crpix_jarron.json'
path = os.path.join(os.path.split(os.path.abspath(__file__))[0], path)
file = open(path, 'r')
crpix_jarron = json.load(file)
file.close()
path = 'resources/filter_shifts_jarron.json'
path = os.path.join(os.path.split(os.path.abspath(__file__))[0], path)
file = open(path, 'r')
filter_shifts_jarron = json.load(file)
file.close()

# Loop through apertures and filters.
for apername in crpix_jarron.keys():
    for filt in filter_shifts_jarron.keys():
        if 'FULL_MASK210R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0038.fits'
            xcen, ycen = 711.5, 1505.5
            cval = 1.
        elif 'MASK210R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0049.fits'
            xcen, ycen = 320.5, 320.5
            cval = 0.99088
        elif 'FULL_MASK335R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0020.fits'
            xcen, ycen = 651.5, 1661.5
            cval = 1.
        elif 'MASK335R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0019.fits'
            xcen, ycen = 160.5, 160.5
            cval = 0.99088
        elif 'FULL_MASK430R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0060.fits'
            xcen, ycen = 973.5, 1661.5
            cval = 1.
        elif 'MASK430R' in apername:
            fitsfile = 'jwst_nircam_psfmask_0027.fits'
            xcen, ycen = 160.5, 160.5
            cval = 0.99088
        
        # Get true mask center from Jarron.
        crpix1_jarron, crpix2_jarron = crpix_jarron[apername]
        print(apername, crpix1_jarron, crpix2_jarron)
        
        # Get filter shift from Jarron.
        xshift_jarron, yshift_jarron = filter_shifts_jarron[filt]
        
        # Compute required shift.
        crpix1, crpix2 = crpix1_jarron + xshift_jarron, crpix2_jarron + yshift_jarron
        xoff, yoff = crpix1 - xcen, crpix2 - ycen
        
        # Read PSF mask.
        hdul = pyfits.open('resources/transmissions/' + fitsfile)
        psfmask = hdul['SCI'].data
        
        # Shift the coronagraphic mask separately with a higher interpolation
        # order.
        xr = np.arange(psfmask.shape[1]) - (xcen - 1.)
        yr = np.arange(psfmask.shape[0]) - (ycen - 1.)
        xx, yy = np.meshgrid(xr, yr)
        rr = np.sqrt(xx**2 + yy**2)
        mask = psfmask.copy()
        mask[rr > 100.] = 0.99088
        mask = shift(mask, (yoff, xoff), order=3, mode='constant', cval=0.99088)
        
        # Shift PSF mask.
        psfmask = shift(psfmask, (yoff, xoff), order=0, mode='constant', cval=cval)
        
        # Insert the separately shifted coronagraphic mask into the full PSF
        # mask.
        xr = np.arange(psfmask.shape[1]) - (crpix1 - 1.)
        yr = np.arange(psfmask.shape[0]) - (crpix2 - 1.)
        xx, yy = np.meshgrid(xr, yr)
        rr = np.sqrt(xx**2 + yy**2)
        psfmask[rr < 100.] = mask[rr < 100.]
        
        # Write PSF mask.
        hdul['SCI'].data = psfmask
        hdul.writeto('resources/transmissions/' + apername + '_' + filt + '.fits', output_verify='fix', overwrite=True)

fitsfiles = ['JWST_MIRI_F1065C_transmission_webbpsf-ext_v2.fits',
             'JWST_MIRI_F1140C_transmission_webbpsf-ext_v2.fits',
             'JWST_MIRI_F1550C_transmission_webbpsf-ext_v2.fits']
for fitsfile in fitsfiles:
    hdul = pyfits.open('resources/transmissions/' + fitsfile)
    psfmask = hdul[0].data
    psfmask[psfmask < 0.05] = np.nan
    hdu = pyfits.ImageHDU(psfmask)
    hdu.header['EXTNAME'] = 'SCI'
    hdul.append(hdu)
    hdul.writeto('resources/transmissions/' + fitsfile, output_verify='fix', overwrite=True)
