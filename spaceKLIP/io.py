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
import scipy.ndimage.interpolation as sinterp
import astropy.units as u
from synphot import SourceSpectrum
from synphot.models import Empirical1D

import webbpsf, webbpsf_ext
webbpsf_ext.setup_logging(level='ERROR', verbose=False)

rad2mas = 180./np.pi*3600.*1000.

# Define logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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

    try:
        temp = config['pa_ranges_bar']
        Nval = len(temp)
        pa_ranges_bar = []
        if Nval % 2 != 0:
            raise UserWarning('pa_ranges_bar needs to be list of 2-tuples')
        for i in range(Nval//2):
            pa_ranges_bar += [(float(temp[2*i][1:]), float(temp[2*i+1][:-1]))]
        config['pa_ranges_bar'] = pa_ranges_bar
    except:
        pass

    try:
        temp = config['pa_ranges_fqpm']
        Nval = len(temp)
        pa_ranges_fqpm = []
        if Nval % 2 != 0:
            raise UserWarning('pa_ranges_fqpm needs to be list of 2-tuples')
        for i in range(Nval//2):
            pa_ranges_fqpm += [(float(temp[2*i][1:]), float(temp[2*i+1][:-1]))]
        config['pa_ranges_fqpm'] = pa_ranges_fqpm
    except:
        pass


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
    IS_PSF    = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    SELFREF  = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    APERNAME = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    PIXSCALE = np.empty(Nfitsfiles) # mas
    PIXAR_SR = np.empty(Nfitsfiles) # sr
    RA_REF = np.empty(Nfitsfiles) # deg
    DEC_REF = np.empty(Nfitsfiles) # deg
    ROLL_REF = np.empty(Nfitsfiles) # deg
    HASH = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    for i, file in enumerate(fitsfiles):
        hdul = pyfits.open(file)

        # Primary Header
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
        ### Kim says these are no longer needed, so commenting out (8/12/2022)
        # if ('SGD' in file): # MIRI test data
        #     SUBPXPTS[i] = 5
        # elif ('HD141569' in file): # MIRI test data
        #     SUBPXPTS[i] = 1
        # else:
        SUBPXPTS[i] = head.get('NUMDTHPT', 1)
        # Check for IS_PSF and SELFREF header keywords
        IS_PSF[i] = head.get('IS_PSF', 'NONE') 
        SELFREF[i] = head.get('SELFREF', 'NONE')
        APERNAME[i] = head.get('APERNAME', 'NONE')
        if (INSTRUME[i] == 'NIRCAM'):
            if ('LONG' in DETECTOR[i] or '5' in DETECTOR[i]):
                PIXSCALE[i] = nrc._pixelscale_long*1e3 # mas
            else:
                PIXSCALE[i] = nrc._pixelscale_short*1e3 # mas
        elif (INSTRUME[i] == 'MIRI'):
            PIXSCALE[i] = mir.pixelscale*1e3 # mas
        else:
            raise UserWarning(f'Unknown instrument: {INSTRUME[i]}. Must be either NIRCAM or MIRI.')

        # Science header
        head = hdul['SCI'].header
        try:
            PIXAR_SR[i] = head['PIXAR_SR'] # sr
        except:
            PIXAR_SR[i] = PIXSCALE[i]**2/rad2mas**2 # sr
        RA_REF[i] = head['RA_REF'] # deg
        DEC_REF[i] = head['DEC_REF'] # deg
        ### Kim says these are no longer needed, so commenting out (8/12/2022)
        # if ('SGD' in file): # MIRI test data
        #     ROLL_REF[i] = 0. # deg
        # elif ('HD141569' in file): # MIRI test data
        #     ROLL_REF[i] = file.split('/')[-1].split('_')[2][2:] # deg
        # else:
        # Roll Ref: V3 roll angle at the ref point (N over E)
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

        # Flight data have keywords IS_PSF and SELFREF to identify ref sources.
        # We are going to try that first, then fall back to previous version
        # if the keywords are not found in order to support backwards 
        # compatability of simulated data
        # TODO: Incorporate SELFREF keyword
        isref_i = IS_PSF[ww]
        if isref_i[0] != 'NONE':
            ww_sci = np.where(isref_i == 'False')[0]
            ww_cal = np.where(isref_i == 'True')[0]
        else:
            log.warning("Unable to find IS_PSF keyword.")

            # Science and reference PSFs are identified based on their number of
            # dither positions, assuming that there is no dithering for the
            # science PSFs and dithering for the reference PSFs.
            dpts = SUBPXPTS[ww]
            dpts_unique = np.unique(dpts)

            if ((len(dpts_unique) == 2) and (dpts_unique[0] == 1)):
                ww_sci = np.where(dpts == dpts_unique[0])[0]
                ww_cal = np.where(dpts == dpts_unique[1])[0]
            else:
                raise UserWarning(
                    'Unable to find IS_PSF keyword, so fell back to looking for NUMDTHPT.'
                    '\nScience and reference PSFs are identified based on their'
                    '\nnumber of dither positions, with the assumption of no dithering'
                    '\nfor the science PSFs and small grid dithers for the reference PSFs.'
                )

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
        tab = Table(names=('TYP', 'TARGPROP', 'TARG_RA', 'TARG_DEC', 'READPATT', 'NINTS', 'NGROUPS', 'NFRAMES', 
                    'EFFINTTM', 'RA_REF', 'DEC_REF','ROLL_REF', 'FITSFILE', 'PIXSCALE', 'PA_V3'),  # PA_V3 is replaced by ROLL_REF
                    dtype=('S', 'S', 'f', 'f', 'S', 'i', 'i', 'i', 'f', 'f', 'f', 'f', 'S', 'f', 'f'))
        for j in range(len(ww_sci)):
            tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], 
                         NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], 
                         RA_REF[ww][ww_sci][j], DEC_REF[ww][ww_sci][j], ROLL_REF[ww][ww_sci][j], 
                         fitsfiles[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], ROLL_REF[ww][ww_sci][j])) # Final ROLL_REF is PA_V3 column
        for j in range(len(ww_cal)):
            tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], 
                         NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], 
                         RA_REF[ww][ww_cal][j], DEC_REF[ww][ww_cal][j], ROLL_REF[ww][ww_cal][j], 
                         fitsfiles[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], ROLL_REF[ww][ww_cal][j])) # Final ROLL_REF is PA_V3 column
        meta.obs[HASH_unique[i]] = tab.copy()
        del tab

    if (meta.verbose == True):
        log.info('--> Identified %.0f concatenation(s)' % len(meta.obs))
        for i, key in enumerate(meta.obs.keys()):
            log.info('--> Concatenation %.0f: ' % (i+1)+key)
            print_tab = copy.deepcopy(meta.obs[key])
            print_tab.remove_column('FITSFILE')
            print_tab.pprint(max_lines=100, max_width=1000)

    return meta

def read_spec_file(file):
    """
    Read a spectrum file in format wavelength / Jansky and return
    a synphot SourceSpectrum object

    Parameters
    ----------
    file: str
        file location
    """
    try: 
        # Open file and grab wavelength and flux arrays
        data = np.genfromtxt(file).transpose()
        model_wave = data[0]
        model_flux = data[1]

        # Create a synphot spectrum
        SED = SourceSpectrum(Empirical1D, points=model_wave << u.Unit('micron'), lookup_table=model_flux << u.Unit('Jy'))
        SED.meta['name'] = file.split('/')[-1]
    except:
        raise ValueError("Unable to read in provided file. Ensure format is in two columns with wavelength (microns), flux (Jy)")

    return SED


def get_working_files(meta, runcheck, subdir='RAMPFIT', search='uncal.fits', itype='default'):

    # Add wild card to the start of the search string
    search = '*' + search

    # Figure out where to look for files
    if runcheck:
        #Use an output directory that was just created
        rdir = meta.odir + subdir + '/'
    else:
        if itype == 'default':
            #Use the specified input directory
            rdir = meta.idir
        elif itype == 'bgsci':
            rdir = meta.bg_sci_dir
        elif itype == 'bgref':
            rdir = meta.bg_ref_dir

    # Grab the files
    files = glob.glob(rdir + search)
    if len(files) == 0:
        # Let's look for a subdir
        if os.path.exists(rdir + subdir):
            log.info('Located {} folder within input directory.'.format(subdir))
            rdir += subdir + '/' + search
            files = glob.glob(rdir)

        # If there are still no files, look in output directory
        if (len(files) == 0) and ('/{}/'.format(subdir) not in rdir):
            log.warning('No {} files found in input directory, searching output directory.'.format(search))
            rdir = meta.odir + subdir + '/' + search
            files = glob.glob(rdir)

        if len(files) == 0:
            raise ValueError('Unable to find any {} files in specified input or output directories.'.format(search))

    if meta.verbose:
        log.info('--> Found {} file(s) under: {}'.format(len(files), rdir))

    return np.sort(files)


def sort_data_files(pid, sci_obs, ref_obs, outdir, expid_sci='03106', 
    file_ext='uncal.fits', indir=None, filter=None, coron_mask=None, verbose=False, filename_start='jw'):
    """Create symbolic links to data in MAST data directory
    
    Place science and reference observations of same kind in their
    own sub-directories.

    Given a sequence of science and reference observation IDs, sort
    exposures with different filters into their own directory in some
    output directory location. Assumes data is in MAST download directory
    as defined by `$JWSTDOWNLOAD_OUTDIR` environment variable, unless
    otherwise specified. Creates symbolic links to data so as to not
    unnecessarily use up disk space.

    Parameters
    ==========
    pid : int
        Program ID.
    sci_obs : array-like
        List of observation numbers corresponding to Roll1 and Roll2
    ref_obs : array_like
        List of observations observed as reference stars.
    outdir : str
        Base path to create directories for each filter / mask.
    
    Keyword Args
    ============
    expid : str
        Exposure ID associated with first science observation, as opposed
        to target acquisition and astrometric confirmation images.
    file_ext : str
        File extension (default: 'uncal.fits')
    indir : str or None
        Location of original files. If not set, then searches for MAST
        directory location at $JWSTDOWNLOAD_OUTDIR env variable.
    filename_start : str
        Initial string at start of filenames. This will generally always be the default 'jw',
        but can be overridden if necessary, for instance if dealing with simulated data.
    """

    from astropy.io import fits

    # MAST and raw data directory
    if indir is None:
        mast_dir = os.getenv('JWSTDOWNLOAD_OUTDIR')
        if mast_dir is None:
            raise RuntimeError('Cannot file environment variable: $JWSTDOWNLOAD_OUTDIR')
        indir = os.path.join(mast_dir, f'{pid:05d}/')

    if verbose:
        log.info(f"""Sorting data from program {pid} for {filter} with {coron_mask} mask, 
                    Sci Obs: {sci_obs}\tPSF Reference Obs: {ref_obs}
                    Sorting files with extension {file_ext}
                    from input dir {indir}
                    into output dir {outdir}""")

    # Find all uncal files
    allfiles = np.sort([f for f in os.listdir(indir) if f.endswith(file_ext)])
    if len(allfiles)==0:
        raise RuntimeError(f"Could not find any files ending with {file_ext}")

    # Cycle through each science observation
    for obsid in sci_obs:
        file_start = f'{filename_start}{pid:05d}{obsid:03d}'

        # Get all files in given observation
        files_obs = np.sort([f for f in allfiles if (file_start in f)])

        if len(files_obs)==0:
            raise RuntimeError(f"Could not find any files matching {file_start}")
        # Get the associated exposure IDS
        expid_index = 1+filename_start.count('_')  # Exp ID is usually at index 1, unless there's an extra underscore earlier
        expids_all = np.array([f.split('_')[expid_index] for f in files_obs])

        # Index of where science data starts
        # Assume expid_sci is the first in a sequence of filters
        if( expid_sci is None) or (expid_sci.lower()=='none'):
            expid_start = expids_all[0]
        else:
            expid_start =expid_sci
        istart = np.where(expids_all==expid_start)[0][0]
        for ii in np.arange(istart, len(expids_all)):
            file_path = os.path.join(indir, files_obs[ii])
            hdr = fits.getheader(file_path)

            # Get filter and 
            exp_type = hdr.get('EXP_TYPE')
            filt = hdr.get('FILTER')
            apname = hdr.get('APERNAME')
            if 'MASK' in apname:
                mask_arr = ['MASK335R', 'MASK430R', 'MASKLWB', 'MASK210R', 'MASKSWB']
                for mask in mask_arr:
                    if mask in apname:
                        image_mask = mask
                image_mask_str = '_' + image_mask
            else:
                image_mask_str = image_mask = ''

            # Skip this file if specified filter or coron mask don't match
            if (filter is not None) and (filter!=filt):
                continue
            if (coron_mask is not None) and (coron_mask!=image_mask):
                continue

            # Get filter directory location
            sub_str = filt + image_mask_str
            subdir = os.path.join(outdir, sub_str)
            # Create if it doesn't currently exist
            if not os.path.isdir(subdir):
                log.info(f'Creating directory: {subdir}')
                os.mkdir(subdir)

            # Generate symbolic link to new location
            file_link_path = os.path.join(subdir, files_obs[ii])
            if not os.path.isfile(file_link_path):
                os.symlink(file_path, file_link_path)

            # Cycle through reference files and find everything
            # with the same filter, exp_type, and apname
            if (ref_obs is not None) and isinstance(ref_obs, (list,np.ndarray)):
                for obsid_ref in ref_obs:
                    file_start_ref = f'jw{pid:05d}{obsid_ref:03d}'
                    # Get all files in given observation
                    files_ref = np.sort([f for f in allfiles if (file_start_ref in f)])
                    for fref in files_ref:
                        file_path_ref = os.path.join(indir, fref)
                        hdr_ref = fits.getheader(file_path_ref)
                        # Get filter and 
                        exp_type_ref = hdr_ref.get('EXP_TYPE')
                        filt_ref = hdr_ref.get('FILTER')
                        apname_ref = hdr_ref.get('APERNAME')

                        if (exp_type_ref==exp_type) and (filt==filt_ref) and (apname_ref==apname):
                            # Generate symbolic link to new location
                            file_link_path = os.path.join(subdir, fref)
                            if not os.path.isfile(file_link_path):
                                os.symlink(file_path_ref, file_link_path)
            else:
                log.warning(f'No ref obs specified: meta.ref_obs={ref_obs}')

    log.info(f"Sorting complete for {pid} {coron_mask} {filter}")

def open_new_log_file(fits_file, output_dir, stage_str=None):
    """Create and open a new log file
    
    Parameters
    ==========
    fits_file : str
        Name of input FITS file that will be parsed to create
        name of log file.
    output_dir : str
        Location to save log file.
    stage_str : str or None
        Pipeline stage of interest, such as 'detector1', 'image2', 
        'coron3', etc.
    """

    import logging
    from datetime import datetime

    # Create log file output name

    # Remove directory and drop file exension
    file_base = os.path.basename(fits_file)
    file_base = '_'.join(file_base.split('_')[:-1])

    date_str = datetime.now().isoformat()
    stage_str = '' if stage_str is None else f'_{stage_str}'
    fname = f'{file_base}{stage_str}_{date_str}.log'
    log_file = os.path.join(output_dir, fname)
    # Create empty file
    with open(log_file, 'w') as f:
        pass

    # Add file stream handler append log messages to file
    logger = logging.getLogger()
    fh = logging.FileHandler(log_file, 'a')
    fmt = logging.Formatter('%(asctime)s [%(name)s:%(levelname)s] %(message)s')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger, fh

def close_log_file(logger, file_handler):
    """Remove handler from logger and close log file."""

    logger.removeHandler(file_handler)
    file_handler.close()

def save_fitpsf_images(odir, fitpsf):

    # create best fit FM
    dx = fitpsf.fit_x.bestfit - fitpsf.data_stamp_x_center
    dy = fitpsf.fit_y.bestfit - fitpsf.data_stamp_y_center

    fm_bestfit = fitpsf.fit_flux.bestfit * sinterp.shift(fitpsf.fm_stamp, [dy, dx])
    if fitpsf.padding > 0:
        fm_bestfit = fm_bestfit[fitpsf.padding:-fitpsf.padding, fitpsf.padding:-fitpsf.padding]

    # make residual map
    residual_map = fitpsf.data_stamp - fm_bestfit

    # Create FITS
    pri = pyfits.PrimaryHDU()
    sci = pyfits.ImageHDU(fitpsf.data_stamp, name='SCI')
    mod = pyfits.ImageHDU(fm_bestfit, name='MOD')
    res = pyfits.ImageHDU(residual_map, name='RES')

    hdul = pyfits.HDUList([pri,sci,res,mod])
    hdul.writeto(odir, overwrite=True)

def expand_path(path_string):
    """Expand environment variables or user home directories in a path specification

    Parameters
    ==========
    path_string : str
        Any path string, relative or absolute, optionally containing environment variables.

    """
    return os.path.expanduser(os.path.expandvars(path_string))

