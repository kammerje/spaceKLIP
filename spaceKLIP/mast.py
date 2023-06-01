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

import astropy
import astroquery
from astroquery.mast import Mast

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

def set_params(parameters):
    """
    Utility function for making dicts used in MAST queries.
    
    """
    
    return [{'paramName': p, 'values': v} for p, v in parameters.items()]

def query_coron_datasets(inst,
                         filt,
                         mask=None,
                         kind=None,
                         ignore_cal=True):
    """
    Query MAST to make a summary table of existing JWST coronagraphic datasets.
    
    Parameters
    ----------
    inst : str
        'NIRCam' or 'MIRI'. Required.
    filt : str
        Filter name, like 'F356W'. Required.
    mask : str
        Coronagraph mask name, like 'MASKA335R' or '4QPM_1550'. Optional.
        If provided, must exactly match the keyword value as used in MAST,
        which isn't always what you might expect. In particular, you need to
        explicitly include the "A" for the NIRCam A module in e.g.,
        'MASKA335R'.
    kind : str
        'SCI' for science targets, 'REF' for PSF references, or 'BKG' for
        backgrounds.
    ignore_cal : bool
        Ignore/exclude any category='CAL' calibration programs. This can be
        desirable to ignore the NIRCam coronagraphic flux calibration data
        sets, which otherwise look like science data to this query (programs
        1537 and 1538 for example).
    
    Returns
    -------
    None.
    
    """
    
    # Perform MAST query to find all available datasets for a given
    # filter/occulter.
    if inst.upper() == 'MIRI':
        service = 'Mast.Jwst.Filtered.Miri'
        template = 'MIRI Coronagraphic Imaging'
    else:
        service = 'Mast.Jwst.Filtered.NIRCam'
        template = 'NIRCam Coronagraphic Imaging'
    
    keywords = {
        'filter': [filt],
        'template': [template],
        'productLevel': ['2b'],
    }
    
    if mask is not None:
        keywords['coronmsk'] = [mask]
    if ignore_cal:
        keywords['category'] = ['COM', 'ERS', 'GTO', 'GO']  # but not CAL
    
    # Optional, restrict to one kind of coronagraphic data
    if kind == 'SCI':
        keywords['is_psf'] = ['f']
        keywords['bkgdtarg'] = ['f']
    elif kind == 'REF':
        keywords['is_psf'] = ['t']
        keywords['bkgdtarg'] = ['f']
    elif kind == 'BKG':
        keywords['is_psf'] = ['f']
        keywords['bkgdtarg'] = ['t']
    
    # Method note: we query MAST for much more than we actually need/want, and
    # then filter down afterwards. This is not entirely necessary, but leaves
    # room for future expansion to add options for more verbose output, etc.
    # Currently this works by retrieving all level2b (i.e., cal/calints)
    # files, including all dithers etc., and then trimming to one unique row
    # per observation.
    collist = 'filename, productLevel, bkgdtarg, bstrtime, duration, effexptm, effinttm, exp_type, filter, coronmsk, is_psf, nexposur, nframes, nints, numdthpt, obs_id, obslabel, pi_name, program, subarray, targname, template, title, visit_id, visitsta, vststart_mjd, isRestricted'
    all_columns = False
    
    parameters = {'columns': '*' if all_columns else collist,
                  'filters': set_params(keywords)}
    
    response = Mast.service_request(service, parameters)
    response.sort(keys='bstrtime')
    
    # Summarize the distinct observations conveniently.
    cols_to_keep = ['visit_id', 'filter', 'coronmsk', 'targname', 'obslabel',
                    'duration', 'numdthpt', 'program', 'title', 'pi_name']
    
    summarytable = astropy.table.Table([response.columns[cname].filled() for cname in cols_to_keep],
                                       masked=False,
                                       copy=True)
    
    # Add the initial V to visit ID.
    summarytable['visit_id'] = astropy.table.Column(summarytable['visit_id'], dtype=np.dtype('<U12'))
    for row in summarytable:
        row['visit_id'] = 'V' + row['visit_id']
    
    # Add a summary for which kind of observation each is.
    kind = np.zeros(len(response), dtype='a3')
    kind[:] = 'SCI'  # by default assume SCI, then check for REF and BKG
    kind[response.columns['is_psf'] == 't'] = 'REF'
    kind[response.columns['bkgdtarg'] == 't'] = 'BKG'
    kind[kind == ''] = 'SCI'
    summarytable.add_column(astropy.table.Column(kind), index=2, name='kind')
    
    summarytable.add_column(astropy.table.Column(astropy.time.Time(response['vststart_mjd'], format='mjd').iso, dtype=np.dtype('<U16')),
                            index=1,
                            name='start time')
    
    summarytable = astropy.table.unique(summarytable)
    summarytable.sort(keys='start time')
    
    return summarytable
