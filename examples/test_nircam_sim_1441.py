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

from spaceKLIP import database, coron1pipeline, coron2pipeline, coron3pipeline, pyklippipeline, imagetools, contrast


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    
    # Set the input and output directories and grab the input FITS files.
    input_dir = '../examples/NIRCam_sim_1441/uncal/'
    output_dir = '../examples/NIRCam_sim_1441/spaceklip/'
    fitsfiles = sorted([input_dir + f for f in os.listdir(input_dir) if f.endswith('.fits')])
    
    # Initialize the spaceKLIP database and read the input FITS files.
    Database = database.Database(output_dir=output_dir)
    Database.read_jwst_s012_data(datapaths=fitsfiles,
                                 psflibpaths=None,
                                 bgpaths=None)
    
    # Run the Coron1Pipeline, the Coron2Pipeline, and the Coron3Pipeline.
    # Additional step parameters can be passed using the steps parameter as
    # outlined here:
    # https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
    coron1pipeline.run_obs(database=Database,
                           steps={'saturation': {'n_pix_grow_sat': 1,
                                                 'grow_diagonal': False},
                                  'refpix': {'odd_even_columns': True,
                                             'odd_even_rows': True,
                                             'nlower': 4,
                                             'nupper': 4,
                                             'nleft': 4,
                                             'nright': 4,
                                             'nrow_off': 0,
                                             'ncol_off': 0},
                                  'dark_current': {'skip': True},
                                  'jump': {'rejection_threshold': 4.,
                                           'three_group_rejection_threshold': 4.,
                                           'four_group_rejection_threshold': 4.},
                                  'ramp_fit': {'save_calibrated_ramp': False}},
                           subdir='stage1')
    coron2pipeline.run_obs(database=Database,
                           steps={'outlier_detection': {'skip': False}},
                           subdir='stage2')
    # coron3pipeline.run_obs(database=Database,
    #                        steps={'klip': {'truncate': 100}},
    #                        subdir='stage3')
    
    # Initialize the spaceKLIP image manipulation tools class.
    ImageTools = imagetools.ImageTools(Database)
    
    # Remove the first frame due to reset switch charge delay. Only required
    # for MIRI.
    # ImageTools.remove_frames(index=[0],
    #                          frame=None,
    #                          types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                          subdir='removed')
    
    # Median-subtract each frame to mitigate uncalibrated bias drifts. Only
    # required for NIRCam.
    ImageTools.subtract_median(types=['SCI', 'SCI_TA', 'SCI_BG', 'REF', 'REF_TA', 'REF_BG'],
                               subdir='medsub')
    
    # Crop all frames.
    # ImageTools.crop_frames(npix=1,
    #                        types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                        subdir='cropped')
    
    # Fix bad pixels using custom spaceKLIP routines. Multiple routines can be
    # combined in a custom order by joining them with a + sign.
    # - bpclean: use sigma clipping to find additional bad pixels.
    # - custom: use custom map to find additional bad pixels.
    # - timemed: replace pixels which are only bad in some frames with their
    #            median value from the good frames.
    # - dqmed:   replace bad pixels with the median of surrounding good
    #            pixels.
    # - medfilt: replace bad pixels with an image plane median filter.
    ImageTools.fix_bad_pixels(method='bpclean+timemed+dqmed+medfilt',
                              bpclean_kwargs={'sigclip': 5,
                                              'shift_x': [-1, 0, 1],
                                              'shift_y': [-1, 0, 1]},
                              custom_kwargs={},
                              timemed_kwargs={},
                              dqmed_kwargs={'shift_x': [-1, 0, 1],
                                            'shift_y': [-1, 0, 1]},
                              medfilt_kwargs={'size': 4},
                              subdir='bpcleaned')
    
    # Perform a background subtraction to remove the MIRI glowstick. Only
    # required for MIRI.
    # ImageTools.subtract_background(subdir='bgsub')
    
    # Replace nans.
    ImageTools.replace_nans(cval=0.,
                            types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                            subdir='nanreplaced')
    
    # Use image registration to align all frames in a concatenation to the
    # first science frame in that concatenation.
    ImageTools.align_frames(method='fourier',
                            kwargs={},
                            subdir='aligned')
    
    # Coadd frames.
    # ImageTools.coadd_frames(nframes=10,
    #                         types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
    #                         subdir='coadded')
    
    # Pad all frames.
    ImageTools.pad_frames(npix=160,
                          cval=np.nan,
                          types=['SCI', 'SCI_BG', 'REF', 'REF_BG'],
                          subdir='padded')
    
    # Run the pyKLIP pipeline. Additional parameters for the klip_dataset
    # function can be passed using the kwargs parameter.
    pyklippipeline.run_obs(database=Database,
                           kwargs={'mode': ['ADI', 'RDI', 'ADI+RDI'],
                                   'annuli': [1, 5],
                                   'subsections': [1],
                                   'numbasis': [1, 2, 5, 10, 20, 50, 100],
                                   'algo': 'klip'},
                           subdir='klipsub')
    
    # Initialize the spaceKLIP contrast estimation class.
    Contrast = contrast.Contrast(Database)
    
    # Compute raw contrast.
    Contrast.raw_contrast(subdir='rawcon')
    
    pdb.set_trace()
