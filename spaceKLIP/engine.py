import os, re

import astropy.io.fits as pyfits
import numpy as np
import copy

from astropy.table import Table
from astroquery.svo_fps import SvoFps

import pysiaf
import webbpsf

os.environ['CRDS_PATH'] = '../crds_cache'
os.environ['CRDS_SERVER_URL'] = 'https://jwst-crds.stsci.edu'

from . import io
from . import subtraction
from . import contrast
from . import companion
from . import utils
from . import rampfit
from . import imgprocess

class Meta():
    """
    A meta class to hold information throughout the pipeline process,
    including user inputs.
    """

    def __init__(self):

        return

    def save(self):
        
        # TODO: write function that could save all of the used parameters for
        # debugging purposes
        return

class Pipeline():
    """
    Generic Pipeline Class.
    """

    def __init__(self, config_file):
        """
        Initialize a generic pipeline class by reading a config file and
        saving inputs to a Meta object.
        """

        # Initialize Meta object
        self.meta = Meta()

        # Read in configuration parameters
        if config_file == 'template':
            tempstring = '/../tests/example_config.yaml'
            config_file = os.path.join(os.path.dirname(__file__) + tempstring)
        config = io.read_config(config_file)

        # Assign config parameters to class attributes
        for key in config:
            setattr(self.meta, key, config[key])

        if self.meta.rundirs != None:
            self.meta.rundirs = [self.meta.odir+rdir.replace('/', '')+'/' for rdir in self.meta.rundirs]

        return

class JWST(Pipeline):
    """
    JWST specifc Pipeline class.
    """

    def __init__(self, config_file):
        """
        Initialize a JWST specific Pipeline Class.

        Note: this class only works with NIRCam so far.

        Parameters
        ----------
        config_file : str
            File path of .yaml configuration file.
        """

        # Intialize parent class
        super().__init__(config_file)

        # Assign flags to track which stages have been performed
        self.meta.done_rampfit = False
        self.meta.done_imgprocess = False
        self.meta.done_subtraction = False
        self.meta.done_raw_contrast = False
        self.meta.done_cal_contrast = False
        self.meta.done_companion = False

        # Get properties for JWST
        self.get_jwst_meta()

        return None

    def get_jwst_meta(self):
        """ 
        Define a range of parameters specific to JWST and its instruments.
        """

        # Define telescope and instrument properties
        self.meta.diam = 6.5 # m; primary mirror diameter
        self.meta.iwa = 1. # pix; inner working angle
        self.meta.owa = 150. # pix; outer working angle

        # Ancillary directories
        if not os.path.isdir(self.meta.ancildir):
            self.meta.ancildir = self.meta.odir+'ANCILLARY/'
        self.meta.psfmaskdir = self.meta.ancildir +'psfmasks/'
        self.meta.offsetpsfdir = self.meta.ancildir+'offsetpsfs/'

        # Mean wavelengths and zero points of the JWST filters from the SVO
        # Filter Profile Service
        self.meta.wave = {}
        self.meta.F0 = {}
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRCAM')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.meta.wave[name] = filter_list['WavelengthMean'][i]/1e4*1e-6 # m
            self.meta.F0[name] = filter_list['ZeroPoint'][i] # Jy
        filter_list = SvoFps.get_filter_list(facility='JWST', instrument='MIRI')
        for i in range(len(filter_list)):
            name = filter_list['filterID'][i]
            name = name[name.rfind('.')+1:]
            self.meta.wave[name] = filter_list['WavelengthMean'][i]/1e4*1e-6 # m
            self.meta.F0[name] = filter_list['ZeroPoint'][i] # Jy
        del filter_list

        # PSF position with respect to the NRCA4_MASKSWB and the NRCA5_MASKLWB
        # subarray, respectively, for each NIRCam filter, from pySIAF
        self.siaf = pysiaf.Siaf('NIRCAM')
        self.meta.offset_swb = {filt: self.get_bar_offset_from_siaf(filt, channel='SW')
                                for filt in ['F182M', 'F187N', 'F210M', 'F212N', 'F200W', 'narrow']} # arcsec
        self.meta.offset_lwb = {filt: self.get_bar_offset_from_siaf(filt, channel='LW')
                                for filt in ['F250M', 'F300M', 'F277W', 'F335M', 'F360M', 'F356W', 'F410M', 'F430M', 'F460M', 'F480M', 'F444W', 'narrow']} # arcsec
        del self.siaf

        return

    def get_bar_offset_from_siaf(self, filt, channel='LW'):
        """
        Get bar offset directly from SIAF.
        """
        if channel == 'SW':
            refapername = 'NRCA4_MASKSWB'
            apername = 'NRCA4_MASKSWB_'+filt.upper()
        else: # otherwise default to LW
            refapername = 'NRCA5_MASKLWB'
            apername = 'NRCA5_MASKLWB_'+filt.upper()
        offset_arcsec = np.sqrt((self.siaf.apertures[refapername].V2Ref-self.siaf.apertures[apername].V2Ref)**2+(self.siaf.apertures[refapername].V3Ref-self.siaf.apertures[apername].V3Ref)**2)
        sign = np.sign(self.siaf.apertures[refapername].V2Ref-self.siaf.apertures[apername].V2Ref)

        return sign*offset_arcsec

    def run_all(self, skip_ramp=False, skip_imgproc=False, skip_sub=False, skip_rawcon=False, skip_calcon=False, skip_comps=False):
        """
        Single function to run all pipeline stages in sequence.
        """
        if not skip_ramp:
            self.rampfit()
        if not skip_imgproc:
            self.imgprocess()
        if not skip_sub:
            self.subtraction()
        if not skip_rawcon:
            self.raw_contrast()
        if not skip_calcon:
            self.cal_contrast()
        if not skip_comps:
            self.companions()
        return
    
    def rampfit(self):
        '''
        Wrapper function for ramp fitting stage
        '''
        # Set meta flag to True if not already
        self.meta.done_rampfit = True

        # Run ramp fitting
        ramp = rampfit.stsci_ramp_fitting(self.meta)

        return 

    def imgprocess(self):
        '''
        Wrapper function for image processing stage
        '''
        # Set meta flag to True if not already
        self.meta.done_imgprocess = True
        
        #Run image processing
        img = imgprocess.stsci_image_processing(self.meta)

        return

    def subtraction(self):
        '''
        Wrapper function for subtraction stage
        '''
        # Set meta flag to True if not already
        self.meta.done_subtraction = True

        #Run subtraction
        sub = subtraction.klip_subtraction(self.meta)
        return

    def raw_contrast(self):
        '''
        Wrapper function for raw contrast  stage
        '''
        # Set meta flag to True if not already
        self.meta.done_raw_contrast = True

        #Run raw contrast calculation
        raw_contrast = contrast.raw_contrast_curve(self.meta)
        return

    def cal_contrast(self):
        '''
        Wrapper function for calibrated contrast stage
        '''
        # Set meta flag to True if not already
        self.meta.done_cal_contrast = True
        # Run calibrated contrast calculation
        cal_contrast = contrast.calibrated_contrast_curve(self.meta)
        return

    def companions(self):
        '''
        Wrapper function for companion stage
        '''
        # Set meta flag to True if not already
        self.meta.done_companion = True
        #Run companion photometry / astrometry
        extract_comps = companion.extract_companions(self.meta)
        return
