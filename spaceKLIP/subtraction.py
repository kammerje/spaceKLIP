import glob, os, sys

import numpy as np
import copy

from datetime import date

import pyklip.instruments.JWST as JWST
import pyklip.parallelized as parallelized

from . import io
from . import utils

def klip_subtraction(meta):
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

    files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS', search=meta.sub_ext)

    # Run some preparation steps on the meta object
    meta = utils.prepare_meta(meta, files)

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

                # Increment counter
                counter += 1
    
    return

