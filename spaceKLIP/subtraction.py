import glob, os, sys

import numpy as np
import copy

from datetime import date
from astropy.io import fits
import matplotlib.pyplot as plt

import pyklip.instruments.JWST as JWST
import pyklip.parallelized as parallelized

from . import io
from . import utils


def perform_subtraction(meta):
    '''
    Perform the PSF subtraction.

    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from
        engine.py.

    '''

    files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS', search=meta.sub_ext)
    # Run some preparation steps on the meta object
    meta = utils.prepare_meta(meta, files)

    if meta.bgsub != 'None':
        print('WARNING: Background subtraction only works if running one filter at a time!')
        bgout = bg_subtraction(meta)
        # Reinitialise meta info for pyKLIP
        files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS+BGSUB', search=meta.sub_ext)
        meta = utils.prepare_meta(meta, files)

    # Perform KLIP subtraction
    klip_subtraction(meta, files)

def bg_subtraction(meta):
    '''
    Perform background subtraction on the processed images
    '''
    if meta.bgsub == 'default' or meta.bgsub == 'pyklip':
        bgsci_files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS_BGSCI', search=meta.sub_ext)
        bgref_files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS_BGREF', search=meta.sub_ext)
    elif meta.bgsub == 'saved':
        # Use saved files
        return None
    elif meta.bgsub == 'sci':
        raise ValueError('Not implemented yet!')
    elif meta.bgsub == 'ref':
        raise ValueError('Not implemented yet!')
    else:
        if meta.bgsub == 'None':
            print('Skipping background subtraction')
        else:
            raise ValueError('Background subtraction {} not recognised'.format(meta.bgsub))

    # Trim the first integration if requested.
    if meta.bgtrim == 'first':
        data_start = 1
    else:
        data_start = 0

    types = ['SCI', 'CAL']
    for i, files in enumerate([bgsci_files, bgref_files]):
        
        bg_sub = median_bg(files, data_start=data_start)

        #Save the median_bg
        primary = fits.PrimaryHDU(bg_sub)
        hdul = fits.HDUList([primary])
        meddir = meta.ancildir + 'median_bg/'
        if not os.path.exists(meddir):
            os.makedirs(meddir)
        hdul.writeto(meddir+'{}.fits'.format(types[i]), overwrite=meta.overwrite)

        for key in meta.obs.keys():
            ww = np.where(meta.obs[key]['TYP'] == types[i])[0]
            filepaths = np.array(meta.obs[key]['FITSFILE'][ww], dtype=str).tolist()

        for file in filepaths:
            with fits.open(file) as hdul:
                data = hdul['SCI'].data
                data -= bg_sub[None,:,:]

                hdul['SCI'].data = data[data_start:]
                if data_start != 0:
                    hdul[0].header['NINTS'] -= data_start 
                    for ext in ['ERR', 'DQ', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:  
                        temp_data = hdul[ext].data
                        hdul[ext].data = np.array(temp_data[data_start:])

                savedir = '/'.join(file.split('/')[:-1]).replace('IMGPROCESS', 'IMGPROCESS+BGSUB')
                savefile = file.split('/')[-1].replace('calints', 'bg_calints')
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                hdul.writeto(savedir+'/'+savefile, overwrite=meta.overwrite)

    return


def median_bg(files, data_start=0):
    '''
    Take a list of bg files and median combine them
    '''

    # Get some information on the array shape
    head = fits.getheader(files[0])
    nints = head['NINTS']
    xdim = head['SUBSIZE1']
    ydim = head['SUBSIZE2']

    # Create array to hold backgrounds
    bg_data = np.empty((len(files)*nints,ydim,xdim))

    # Loop over files and extract background
    for i, file in enumerate(files):
        start = i*nints + data_start
        end = (i+1)*nints

        # Grab the data
        if i != len(files)-1:
            bg_data[start:end] = fits.getdata(file, 'SCI')[data_start:]
        else:
            bg_data[start:] = fits.getdata(file, 'SCI')[data_start:]

    # Take a median of the data
    bg_median = np.nanmedian(bg_data, axis=0)

    # TODO Add outlier cleanin

    return bg_median

def klip_subtraction(meta, files):
    """
    Run pyKLIP.

    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from
        engine.py.
    """

    if meta.verbose:
        print('--> Running pyKLIP...')

    # Loop through all modes, numbers of annuli, and numbers of subsections.
    Nscenarios = len(meta.mode)*len(meta.annuli)*len(meta.subsections)
    counter = 1
    meta.truenumbasis = {}
    meta.rundirs = [] # create an array to save the run directories to
    for mode in meta.mode:
        for annuli in meta.annuli:
            for subsections in meta.subsections:

                # Update terminal if requested
                if meta.verbose:
                    print('--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))

                # Create an output directory for each set of pyKLIP parameters
                today = date.today().strftime('%Y_%m_%d_')
                odir = meta.odir+today+mode+'_annu{}_subs{}_run'.format(annuli, subsections)

                # Figure out how many runs of this type have already been
                # performed
                existing_runs = glob.glob(odir+'*'.format(today))

                # Assign run number based on existing runs
                odir += str(len(existing_runs)+1)+'/'

                # Save the odir to the meta object for later analyses
                meta.rundirs.append(odir)

                # Now provide and create actual directory to save to
                odir += 'SUBTRACTED/'
                if not os.path.exists(odir):
                    os.makedirs(odir)

                # Loop through all sets of observing parameters. Only run
                # pyKLIP if the corresponding KLmodes-all fits file does
                # not exist yet.
                for i, key in enumerate(meta.obs.keys()):
                    meta.truenumbasis[key] = [num for num in meta.numbasis if (num <= meta.maxnumbasis[key])]
                    if meta.overwrite == False and os.path.exists(odir+'-KLmodes-all.fits'):
                        continue
                    ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]
                    filepaths = np.array(meta.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                    ww_cal = np.where(meta.obs[key]['TYP'] == 'CAL')[0]
                    psflib_filepaths = np.array(meta.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                    dataset = JWST.JWSTData(filepaths=filepaths,
                                            psflib_filepaths=psflib_filepaths, centering=meta.centering_alg, badpix_threshold=meta.badpix_threshold,
                                            scishiftfile=meta.ancildir+'scishifts', refshiftfile=meta.ancildir+'refshifts')

                    parallelized.klip_dataset(dataset=dataset,
                                              mode=mode,
                                              outputdir=odir,
                                              fileprefix=key,
                                              annuli=annuli,
                                              subsections=subsections,
                                              movement=1,
                                              numbasis=meta.truenumbasis[key],
                                              calibrate_flux=False,
                                              maxnumbasis=meta.maxnumbasis[key],
                                              psf_library=dataset.psflib,
                                              highpass=False,
                                              verbose=meta.verbose)

                # Save a meta file under each directory
                smeta = copy.deepcopy(meta)
                smeta.used_mode = mode
                smeta.used_annuli = annuli
                smeta.used_subsections = subsections

                io.meta_to_json(smeta, savefile=odir+'MetaSave.json')

                # Increment counter
                counter += 1
    
    return
