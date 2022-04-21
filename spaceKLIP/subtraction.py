import glob, os
import sys
from datetime import date
import numpy as np
from astropy.table import Table

import pyklip.instruments.JWST as JWST
import pyklip.parallelized as parallelized

import copy

from . import io

def klip_subtraction(meta):
    """
    Run pyKLIP.
    
    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from engine.py

    Returns
    -------
    TBD

    """ 
    if (meta.verbose == True):
        print('--> Running pyKLIP...')

    search = '*' + meta.sub_ext

    # Figure out where to look for files
    if meta.do_imgprocess:
        #Use the output directory that was just created
        rdir = meta.odir + 'IMGPROCESS/'
    else:
        #Use the specified input directory
        rdir = meta.idir

    # Grab the files
    files = glob.glob(rdir + search)
    if len(files) == 0:
        # Let's look for a 'IMGPROCESS' subdir
        if os.path.exists(rdir + 'IMGPROCESS'):
            print('Located IMGPROCESS folder within input directory.')
            rdir += 'IMGPROCESS/' + search
            files = glob.glob(rdir)
        
        # If there are still no files, look in output directory
        if (len(files) == 0) and ('/IMGPROCESS/' not in rdir):
            print('WARNING: No {} files found in input directory, searching output directory.'.format(search))
            rdir = meta.odir + 'IMGPROCESS/' + search
            files = glob.glob(rdir)

        if len(files) == 0:
            raise ValueError('Unable to find any {} files in specified input or output directories.'.format(search))

    if meta.verbose:
        print('Found {} files under: {}'.format(len(files), rdir))

    
    # Loop through all modes, numbers of annuli, and numbers of subsections.
    Nscenarios = len(meta.mode)*len(meta.annuli)*len(meta.subsections)
    counter = 1
    meta.truenumbasis = {}
    meta.rundirs = [] # Create an array to save the run directories to
    for mode in meta.mode:
        for annuli in meta.annuli:
            for subsections in meta.subsections:
                # Update terminal if requested
                if (meta.verbose == True):
                    sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f \n' % (annuli, subsections, counter, Nscenarios))
                    sys.stdout.flush()
                
                ### Create an output directory for each set of pyKLIP parameters.
                today = date.today().strftime("%Y_%m_%d_")
                odir = meta.odir+today+mode+'_annu{}_subs{}_run'.format(annuli, subsections)

                # Figure out how many runs of this type have already been performed
                existing_runs = glob.glob(odir+'*'.format(today))

                # Assign run numbers based on existing runs
                odir += str(len(existing_runs)+1)+'/'

                # Save the odir to the meta object for later analyses
                meta.rundirs.append(odir)

                # Now provide an create actual directory to save to
                odir += 'SUBTRACTED/'
                if (not os.path.exists(odir)):
                    os.makedirs(odir)
                
                # Loop through all sets of observing parameters. Only run
                # pyKLIP if the corresponding KLmodes-all fits file does
                # not exist yet.
                for i, key in enumerate(meta.obs.keys()):
                    meta.truenumbasis[key] = [num for num in meta.numbasis if (num <= meta.maxnumbasis[key])]
                    if (meta.overwrite == False) and (os.path.exists(odir+'-KLmodes-all.fits')):
                        continue
                    ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]
                    filepaths = np.array(meta.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                    ww_cal = np.where(meta.obs[key]['TYP'] == 'CAL')[0]
                    psflib_filepaths = np.array(meta.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                    dataset = JWST.JWSTData(filepaths=filepaths,
                                            psflib_filepaths=psflib_filepaths)
                    
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

                #Increment counter
                counter += 1
    
    return 