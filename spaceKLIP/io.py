from __future__ import division


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np
import json, yaml

import glob, os
import copy

from astropy.table import Table
import astropy.io.fits as pyfits

import webbpsf
webbpsf.setup_logging(level='ERROR')

rad2mas = 180./np.pi*3600.*1000.


# =============================================================================
# MAIN
# =============================================================================

def read_config(file):
    """
    Read a .yaml configuration file that defines the code execution.

    Parameters
    ----------
    file : str  
        File path of .yaml configuration file.

    Returns
    -------
    config : dict
        Dictionary of all configuration parameters.
    """

    with open(file, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            raise yaml.YAMLError

    temp = config['pa_ranges_bar']
    Nval = len(temp)
    pa_ranges_bar = []
    if Nval % 2 != 0:
        raise UserWarning('pa_ranges_bar needs to be list of 2-tuples')
    for i in range(Nval//2):
        pa_ranges_bar += [(float(temp[2*i][1:]), float(temp[2*i+1][:-1]))]
    config['pa_ranges_bar'] = pa_ranges_bar

    return config

def meta_to_json(input_meta, savefile='./MetaSave.txt'):
    '''
    Convert meta object to a dictionary and save into a json file
    '''
    meta = copy.deepcopy(input_meta)

    # Remove transmission as it is a function
    meta.transmission = None

    # Need to convert astropy table to strings
    for i in meta.obs:
        meta.obs[i] = Table.pformat_all(meta.obs[i])

    # Need to find numpy types and convert to default python types
    for i in vars(meta).keys():
        # Is it a numpy data type?
        if isinstance(vars(meta)[i], np.generic):
            to_py = getattr(meta, i).item()
            setattr(meta, i, to_py)

        # Check another level down
        if isinstance(vars(meta)[i], dict):
            d = getattr(meta, i)

            for key in d.keys():
                if isinstance(d[key], np.generic):
                    d[key] = d[key].item()
            setattr(meta, i, d)
        elif isinstance(vars(meta)[i], list):
            l = getattr(meta, i)

            for j in l:
                if isinstance(j, np.generic):
                    l[j] = l[j].item()
            setattr(meta, i, l)

    with open(savefile, 'w') as msavefile:
        json.dump(vars(meta), msavefile)

    return

def read_metajson(file):
    """
    Load a Meta save file as a json dictionary, don't convert back into a
    class as it doesn't seem necessary yet.
    """

    with open(file) as f:
        metasave = json.load(f)

    return metasave

def extract_obs(meta, fitsfiles_all):
    """
    Extract the metadata of the observations from the FITS files. All
    observations with the same hash (i.e., INSTRUME, DETECTOR, FILTER, PUPIL,
    CORONMSK, SUBARRAY, APERNAME) will be grouped together into a
    concatenation. Each concatenation will then be reduced separately by
    spaceKLIP. Science and reference PSFs are identified based on their number
    of dither positions, assuming that there is no dithering for the science
    PSFs and dithering for the reference PSFs.
    
    Parameters
    ----------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    fitsfiles_all : list of str
        List of the FITS files whose metadata shall be extracted.
    
    Returns
    -------
    meta : object of type meta
        Meta object that contains all the metadata of the observations.
    
    """
    
    # Get all FITS files whose exposure type is compatible with spaceKLIP.
    fitsfiles = []
    for file in fitsfiles_all:
        if (pyfits.getheader(file)['EXP_TYPE'] in ['NRC_IMAGE', 'NRC_CORON', 'MIR_IMAGE', 'MIR_LYOT', 'MIR_4QPM']):
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
        hdul = pyfits.open(file)
        
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
                SUBPXPTS[i] = head['SUBPXPTS']
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
    
    # Group together all observations with the same hash into a concatenation.
    HASH_unique = np.unique(HASH)
    NHASH_unique = len(HASH_unique)
    meta.instrume = {}
    meta.detector = {}
    meta.filter = {}
    meta.pupil = {}
    meta.coronmsk = {}
    meta.subarray = {}
    meta.apername = {}
    meta.pixscale = {}
    meta.pixar_sr = {}
    meta.obs = {}
    for i in range(NHASH_unique):
        ww = HASH == HASH_unique[i]
        
        # Science and reference PSFs are identified based on their number of
        # dither positions, assuming that there is no dithering for the
        # science PSFs and dithering for the reference PSFs.
        dpts = SUBPXPTS[ww]
        dpts_unique = np.unique(dpts)
        try:
            ww_sci = []
            for j in range(len(meta.sci)):
                ww_sci += [np.where(fitsfiles == meta.idir+meta.sci[j])[0][0]]
            ww_sci = np.array(ww_sci)
            ww_cal = []
            for j in range(len(meta.cal)):
                ww_cal += [np.where(fitsfiles == meta.idir+meta.cal[j])[0][0]]
            ww_cal = np.array(ww_cal)
            if ((len(ww_sci) == 0) or (len(ww_cal) == 0)):
                raise UserWarning('No science or calibrator data found')
        except:
            if ((len(dpts_unique) == 2) and (dpts_unique[0] == 1)):
                ww_sci = np.where(dpts == dpts_unique[0])[0]
                ww_cal = np.where(dpts == dpts_unique[1])[0]
            else:
                raise UserWarning('Science and reference PSFs are identified based on their number of dither positions, assuming that there is no dithering for the science PSFs and dithering for the reference PSFs')
        
        # These metadata are the same for all observations within one
        # concatenation.
        meta.instrume[HASH_unique[i]] = INSTRUME[ww][ww_sci][0]
        meta.detector[HASH_unique[i]] = DETECTOR[ww][ww_sci][0]
        meta.filter[HASH_unique[i]] = FILTER[ww][ww_sci][0]
        meta.pupil[HASH_unique[i]] = PUPIL[ww][ww_sci][0]
        meta.coronmsk[HASH_unique[i]] = CORONMSK[ww][ww_sci][0]
        meta.subarray[HASH_unique[i]] = SUBARRAY[ww][ww_sci][0]
        meta.apername[HASH_unique[i]] = APERNAME[ww][ww_sci][0]
        meta.pixscale[HASH_unique[i]] = PIXSCALE[ww][ww_sci][0]
        meta.pixar_sr[HASH_unique[i]] = PIXAR_SR[ww][ww_sci][0]
        
        # These metadata are different for each observation within the
        # concatenation.
        # TODO: PIXSCALE and PA_V3 will be removed in a future version because
        #       they are duplicates. They are kept for now to ensure backward
        #       compatibility.
        tab = Table(names=('TYP', 'TARGPROP', 'TARG_RA', 'TARG_DEC', 'READPATT', 'NINTS', 'NGROUPS', 'NFRAMES', 'EFFINTTM', 'RA_REF', 'DEC_REF', 'ROLL_REF', 'FITSFILE', 'PIXSCALE', 'PA_V3'), dtype=('S', 'S', 'f', 'f', 'S', 'i', 'i', 'i', 'f', 'f', 'f', 'f', 'S', 'f', 'f'))
        for j in range(len(ww_sci)):
            tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], RA_REF[ww][ww_sci][j], DEC_REF[ww][ww_sci][j], ROLL_REF[ww][ww_sci][j], fitsfiles[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], ROLL_REF[ww][ww_sci][j]))
        for j in range(len(ww_cal)):
            tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], RA_REF[ww][ww_cal][j], DEC_REF[ww][ww_cal][j], ROLL_REF[ww][ww_cal][j], fitsfiles[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], ROLL_REF[ww][ww_cal][j]))
        meta.obs[HASH_unique[i]] = tab.copy()
        del tab
    
    temp = meta.obs.copy()
    meta.obs = {}
    for i, key in enumerate(temp.keys()):
        if (i in [0]):
            meta.obs[key] = temp[key].copy()
    
    if (meta.verbose == True):
        print('--> Identified %.0f concatenation(s)' % len(meta.obs))
        for i, key in enumerate(meta.obs.keys()):
            print('--> Concatenation %.0f: ' % (i+1)+key)
            print_tab = copy.deepcopy(meta.obs[key])
            print_tab.remove_column('FITSFILE')
            print_tab.pprint(max_lines=100, max_width=1000)
    
    return meta

def get_working_files(meta, runcheck, subdir='RAMPFIT', search='uncal.fits'):

    # Add wild card to the start of the search string
    search = '*' + search 
   
    # Figure out where to look for files
    if runcheck:
        #Use an output directory that was just created
        rdir = meta.odir + subdir + '/'
    else:
        #Use the specified input directory
        rdir = meta.idir

    # Grab the files
    files = glob.glob(rdir + search)
    if len(files) == 0:
        # Let's look for a subdir
        if os.path.exists(rdir + subdir):
            print('Located {} folder within input directory.'.format(subdir))
            rdir += subdir + '/' + search
            files = glob.glob(rdir)
        
        # If there are still no files, look in output directory
        if (len(files) == 0) and ('/{}/'.format(subdir) not in rdir):
            print('WARNING: No {} files found in input directory, searching output directory.'.format(search))
            rdir = meta.odir + subdir + '/' + search
            files = glob.glob(rdir)

        if len(files) == 0:
            raise ValueError('Unable to find any {} files in specified input or output directories.'.format(search))

    if meta.verbose:
        print('--> Found {} file(s) under: {}'.format(len(files), rdir))

    return files
