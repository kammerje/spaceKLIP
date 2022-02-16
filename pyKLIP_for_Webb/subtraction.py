import os
import sys

import numpy as np

import pyklip.instruments.JWST as JWST
import pyklip.parallelized as parallelized

def KLIPSubtraction(obs, params):
    """
    Run pyKLIP.
    
    Parameters
    ----------
    obs : dict
        Dictionary containing the observation configurations of the input files. 
    params : object
        engine.Params() object containing all configuration parameters for pipeline
        execution

    Returns
    -------
    TBD

    """ 
    verbose = params.verbose

    if (verbose == True):
        print('--> Running pyKLIP...')
    
    # Loop through all modes, numbers of annuli, and numbers of
    # subsections.
    Nscenarios = len(params.mode)*len(params.annuli)*len(params.subsections)
    counter = 1
    params.truenumbasis = {}
    for mode in params.mode:
        for annuli in params.annuli:
            for subsections in params.subsections:
                
                if (verbose == True):
                    sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                    sys.stdout.flush()
                
                # Create an output directory for each set of pyKLIP
                # parameters.
                odir = params.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                if (not os.path.exists(odir)):
                    os.makedirs(odir)
                
                # Loop through all sets of observing parameters. Only run
                # pyKLIP if the corresponding KLmodes-all fits file does
                # not exist yet.
                for i, key in enumerate(obs.keys()):
                    params.truenumbasis[key] = [num for num in params.numbasis if (num <= params.maxnumbasis[key])]
                    if (os.path.exists(odir+key+'-KLmodes-all.fits')):
                        continue
                    ww_sci = np.where(obs[key]['TYP'] == 'SCI')[0]
                    filepaths = np.array(obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                    ww_cal = np.where(obs[key]['TYP'] == 'CAL')[0]
                    psflib_filepaths = np.array(obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                    dataset = JWST.JWSTData(filepaths=filepaths,
                                            psflib_filepaths=psflib_filepaths)
                    
                    parallelized.klip_dataset(dataset=dataset,
                                              mode=mode,
                                              outputdir=odir,
                                              fileprefix=key,
                                              annuli=annuli,
                                              subsections=subsections,
                                              movement=1,
                                              numbasis=params.truenumbasis[key],
                                              calibrate_flux=False,
                                              maxnumbasis=params.maxnumbasis[key],
                                              psf_library=dataset.psflib,
                                              highpass=False,
                                              verbose=False)
                counter += 1
    
    if (verbose == True):
        print('')
    
    return None