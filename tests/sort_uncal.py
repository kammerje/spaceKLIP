# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import json, yaml
import difflib
import glob, os
import copy
import shutil
from astropy.table import Table
import astropy.io.fits as fits

import webbpsf

rad2mas = 180./np.pi*3600.*1000.

# =============================================================================
# MAIN
# =============================================================================

rdir = '/Users/wbalmer/JWST-HCI/HIP65426/NIRCAM/'

search = '*uncal.fits'

def get_fits_all(rdir, search):

    files = glob.glob(rdir+search)

    return files

def sort_uncal(fitsfiles_all):

    # Get all FITS files whose exposure type is compatible with spaceKLIP.
    fitsfiles = []
    for file in fitsfiles_all:
        if (fits.getheader(file)['EXP_TYPE'] in ['NRC_IMAGE', 'NRC_CORON', 'MIR_IMAGE', 'MIR_LYOT', 'MIR_4QPM']):
            fitsfiles += [file]
    fitsfiles = np.array(fitsfiles)

    # Load the WebbPSF NIRCam and MIRI classes.
    nrc = webbpsf.NIRCam()
    mir = webbpsf.MIRI()

    # Extract the metadata of the observations from the FITS files.
    Nfitsfiles = len(fitsfiles)
    TARGPROP = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    TARG_RA = np.empty(Nfitsfiles) # deg
    TARG_DEC = np.empty(Nfitsfiles) # deg
    INSTRUME = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    DETECTOR = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    FILTER = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    PUPIL = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    CORONMSK = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    READPATT = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    NINTS = np.empty(Nfitsfiles, dtype=int)
    NGROUPS = np.empty(Nfitsfiles, dtype=int)
    NFRAMES = np.empty(Nfitsfiles, dtype=int)
    EFFINTTM = np.empty(Nfitsfiles) # s
    SUBARRAY = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    SUBPXPTS = np.empty(Nfitsfiles, dtype=int)
    APERNAME = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    PIXSCALE = np.empty(Nfitsfiles) # mas
    PIXAR_SR = np.empty(Nfitsfiles) # sr
    RA_REF = np.empty(Nfitsfiles) # deg
    DEC_REF = np.empty(Nfitsfiles) # deg
    ROLL_REF = np.empty(Nfitsfiles) # deg
    HASH = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    for i, file in enumerate(fitsfiles):
        hdul = fits.open(file)

        head = hdul[0].header
        if ('SGD' in file): # MIRI test data
            TARGPROP[i] = 'CALIBRATOR'
        elif ('HD141569' in file): # MIRI test data
            TARGPROP[i] = 'HD141569'
        else:
            TARGPROP[i] = head['TARGPROP']
        TARG_RA[i] = head['TARG_RA'] # deg
        TARG_DEC[i] = head['TARG_DEC'] # deg
        INSTRUME[i] = head['INSTRUME']
        DETECTOR[i] = head['DETECTOR']
        FILTER[i] = head['FILTER']
        try:
            PUPIL[i] = head['PUPIL']
        except:
            PUPIL[i] = 'NONE'
        try:
            CORONMSK[i] = head['CORONMSK']
        except:
            CORONMSK[i] = 'NONE'
        READPATT[i] = head['READPATT']
        NINTS[i] = head['NINTS']
        NGROUPS[i] = head['NGROUPS']
        NFRAMES[i] = head['NFRAMES']
        EFFINTTM[i] = head['EFFINTTM'] # s
        SUBARRAY[i] = head['SUBARRAY']
        if ('SGD' in file): # MIRI test data
            SUBPXPTS[i] = 5
        elif ('HD141569' in file): # MIRI test data
            SUBPXPTS[i] = 1
        else:
            try:
                SUBPXPTS[i] = head['NUMDTHPT']
            except:
                SUBPXPTS[i] = 1
            try:
                SUBPXPTS[i] = head['NUMDTHPT']
            except:
                SUBPXPTS[i] = 1
        try:
            APERNAME[i] = head['APERNAME']
        except:
            APERNAME[i] = 'NONE'
        if (INSTRUME[i] == 'NIRCAM'):
            if ('LONG' in DETECTOR[i]):
                PIXSCALE[i] = nrc._pixelscale_long*1e3 # mas
            else:
                PIXSCALE[i] = nrc._pixelscale_short*1e3 # mas
        elif (INSTRUME[i] == 'MIRI'):
            PIXSCALE[i] = mir.pixelscale*1e3 # mas
        else:
            raise UserWarning('Unknown instrument')

        head = hdul['SCI'].header
        try:
            PIXAR_SR[i] = head['PIXAR_SR'] # sr
        except:
            PIXAR_SR[i] = PIXSCALE[i]**2/rad2mas**2 # sr
        RA_REF[i] = head['RA_REF'] # deg
        DEC_REF[i] = head['DEC_REF'] # deg
        if ('SGD' in file): # MIRI test data
            ROLL_REF[i] = 0. # deg
        elif ('HD141569' in file): # MIRI test data
            ROLL_REF[i] = file.split('/')[-1].split('_')[2][2:] # deg
        else:
            ROLL_REF[i] = head['ROLL_REF'] # deg

        # Create a hash for each observation. All observations with the same
        # hash will be grouped together into a concatenation. Each
        # concatenation will then be reduced separately by spaceKLIP.
        HASH[i] = INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]+'_'+APERNAME[i]

        hdul.close()
    HASH_unique = np.unique(HASH)
    NHASH_unique = len(HASH_unique)

    hash0 = HASH_unique[0]
    for hash in HASH_unique:
        hash_filter = hash.split('_')[2]
        print(hash_filter)
        filter_dir = rdir+hash_filter+'/'
        try:
            os.makedirs(filter_dir)
        except FileExistsError:
            continue
        for i, file in enumerate(fitsfiles):
            hdul = fits.open(file)
            head = hdul[0].header
            if head['FILTER'] == hash_filter:
                shutil.copy(file, filter_dir)

            hdul.close()



    return

if __name__ == '__main__':
    sort_uncal(get_fits_all(rdir, search))
