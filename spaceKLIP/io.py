from __future__ import division

import yaml
import json
import glob, os
import copy

import numpy as np
from astropy.table import Table
import astropy.io.fits as pyfits

def read_config(file):
    """
    Read a .yaml configuration file that defines the code execution. 

    Parameters
    ----------
    file : str  
        File path of .yaml configuration file

    Returns
    -------
    config : dict
        Dictionary of all configuration parameters

    """

    with open(file, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except:
            raise yaml.YAMLError

    temp = config['pa_ranges_bar']
    Nval = len(temp)
    pa_ranges_bar = []
    if (Nval % 2 != 0):
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


def read_metajson(filepath):
    '''
    Load a Meta save file as a json dictionary, don't convert
    back into a class as it doesn't seem necessary yet. 
    '''
    with open(filepath) as f:
        metasave = json.load(f)

    return metasave


def extract_obs(meta, fitsfiles_all):
    # Create an astropy table for each unique set of observing parameters
    # (filter, coronagraph, ...). Save all information that is needed
    # later into this table. Finally, save all astropy tables into a
    # dictionary called meta.obs.

    fitsfiles = []
    for file in fitsfiles_all:
        if pyfits.getheader(file)['EXP_TYPE'] in ['NRC_IMAGE', 'NRC_CORON', 'MIR_IMAGE', 'MIR_LYOT', 'MIR_4QPM']:
            fitsfiles += [file]
    fitsfiles = np.array(fitsfiles)

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
    PIXSCALE = np.empty(Nfitsfiles) # mas
    PA_V3 = np.empty(Nfitsfiles) # deg
    HASH = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
    for i, file in enumerate(fitsfiles):
        hdul = pyfits.open(file)
        head = hdul[0].header

        TARGPROP[i] = head['TARGPROP']
        TARG_RA[i] = head['TARG_RA'] # deg
        TARG_DEC[i] = head['TARG_DEC'] # deg
        INSTRUME[i] = head['INSTRUME']
        DETECTOR[i] = head['DETECTOR']
        FILTER[i] = head['FILTER']
        PUPIL[i] = head['PUPIL']
        CORONMSK[i] = head['CORONMSK']
        READPATT[i] = head['READPATT']
        NINTS[i] = head['NINTS']
        NGROUPS[i] = head['NGROUPS']
        NFRAMES[i] = head['NFRAMES']
        EFFINTTM[i] = head['EFFINTTM'] # s
        SUBARRAY[i] = head['SUBARRAY']
        try:
            SUBPXPTS[i] = head['SUBPXPTS']
        except:
            SUBPXPTS[i] = 1
        if ('LONG' in DETECTOR[i]):
            PIXSCALE[i] = meta.pxsc_lw # mas
        else:
            PIXSCALE[i] = meta.pxsc_sw # mas
        head = hdul[1].header
        PA_V3[i] = head['PA_V3'] # deg
        HASH[i] = INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]
        hdul.close()

    HASH_unique = np.unique(HASH)
    NHASH_unique = len(HASH_unique)
    meta.obs = {}
    for i in range(NHASH_unique):
        ww = HASH == HASH_unique[i]
        dpts = SUBPXPTS[ww]
        dpts_unique = np.unique(dpts)
        if ((len(dpts_unique) == 2) and (dpts_unique[0] == 1)):
            ww_sci = np.where(dpts == dpts_unique[0])[0]
            ww_cal = np.where(dpts == dpts_unique[1])[0]
        else:
            raise UserWarning('Science and reference PSFs are identified based on their number of dither positions, assuming that there is no dithering for the science PSFs')
        tab = Table(names=('TYP', 'TARGPROP', 'TARG_RA', 'TARG_DEC', 'READPATT', 'NINTS', 'NGROUPS', 'NFRAMES', 'EFFINTTM', 'PIXSCALE', 'PA_V3', 'FITSFILE'), dtype=('S', 'S', 'f', 'f', 'S', 'i', 'i', 'i', 'f', 'f', 'f', 'S'))
        for j in range(len(ww_sci)):
            tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], PA_V3[ww][ww_sci][j], fitsfiles[ww][ww_sci][j]))
        for j in range(len(ww_cal)):
            tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], PA_V3[ww][ww_cal][j], fitsfiles[ww][ww_cal][j]))
        meta.obs[HASH_unique[i]] = tab.copy()
    
    if (meta.verbose == True):
        print('--> Identified %.0f observation sequences' % len(meta.obs))
        for i, key in enumerate(meta.obs.keys()):
            print('--> Sequence %.0f: ' % (i+1)+key)
            print_table = copy.deepcopy(meta.obs[key])
            print_table.remove_column('FITSFILE')
            print_table.pprint(max_lines=100, max_width=1000)

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
        print('Found {} files under: {}'.format(len(files), rdir))

    return files
