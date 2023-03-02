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

from jwst import datamodels
from jwst.associations.load_as_asn import LoadAsLevel2Asn
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class Coron2Pipeline_spaceKLIP(Image2Pipeline):
    """
    The spaceKLIP JWST stage 2 pipeline class.
    """
    
    def __init__(self,
                 **kwargs):
        
        # Add outlier detection.
        self.step_defs['outlier_detection'] = OutlierDetectionStep
        
        # Initialize Image2Pipeline class.
        super(Coron2Pipeline_spaceKLIP, self).__init__(**kwargs)
        
        # Set additional step parameters.
        self.outlier_detection.skip = False
        
        pass
    
    def process(self,
                input):
        
        # Open input as asn model.
        asn = LoadAsLevel2Asn.load(input, basename=self.output_file)
        
        # Loop through all products.
        all_res = []
        for product in asn['products']:
            if self.save_results:
                self.output_file = product['name']
            try:
                getattr(asn, 'filename')
            except AttributeError:
                asn.filename = 'singleton'
            
            # Process exposure.
            filebase = os.path.basename(asn.filename)
            res = self.process_exposure_product(product, asn['asn_pool'], filebase)
            
            # Run outlier detection.
            res = self.outlier_detection.run(res)
            
            # Save results.
            suffix = 'cal'
            if isinstance(res, datamodels.CubeModel):
                suffix = 'calints'
            res.meta.filename = self.make_output_path(suffix=suffix)
            all_res.append(res)
        
        # Setup output file.
        self.output_use_model = True
        self.suffix = False
        
        return all_res

def run_obs(Database,
            steps={},
            subdir='stage2'):
    
    # Set output directory.
    output_dir = os.path.join(Database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through concatenations.
    for i, key in enumerate(Database.obs.keys()):
        log.info('--> Concatenation ' + key)
        
        # Loop through FITS files.
        Nfitsfiles = len(Database.obs[key])
        for j in range(Nfitsfiles):
            
            # Skip non-stage 1 files.
            head, tail = os.path.split(Database.obs[key]['FITSFILE'][j])
            if Database.obs[key]['DATAMODL'][j] != 'STAGE1':
                log.info('  --> Coron2Pipeline: skipping non-stage 1 file ' + tail)
                continue
            log.info('  --> Coron2Pipeline: processing ' + tail)
            
            # Initialize Coron2Pipeline.
            pipeline = Coron2Pipeline_spaceKLIP(output_dir=output_dir)
            pipeline.save_results = True
            
            # Set step parameters.
            for key1 in steps.keys():
                for key2 in steps[key1].keys():
                    setattr(getattr(pipeline, key1), key2, steps[key1][key2])
            
            # Run Coron2Pipeline.
            fitspath = os.path.abspath(Database.obs[key]['FITSFILE'][j])
            res = pipeline.run(fitspath)
            if isinstance(res, list):
                res = res[0]
            
            # Update spaceKLIP database.
            fitsfile = os.path.join(output_dir, res.meta.filename)
            if fitsfile.endswith('cal.fits'):
                if os.path.isfile(fitsfile.replace('cal.fits', 'calints.fits')):
                    fitsfile = fitsfile.replace('cal.fits', 'calints.fits')
            Database.update_obs(key, j, fitsfile)
    
    pass
