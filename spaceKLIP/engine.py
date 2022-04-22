import os
import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table
import copy

from . import io
from . import subtraction
from . import contrast
from . import companion
from . import utils
from . import rampfit
from . import imgprocess

class Meta():
    """
    A meta class to hold information throughout the pipeline process, including user inputs. 
    """
    def __init__(self):
        return

    def save(self):
        # TODO, write function that could save all of the used parameters
        # for debugging purposes.
        return

class Pipeline():
    """
    Generic Pipeline Class
    """
    def __init__(self, config_file):
        """
        Initialise a generic pipeline class by reading a config file and saving inputs
        to a Meta object. 
        """

        # Initialise Params object
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
            self.meta.rundirs = [self.meta.odir+rdir.replace('/','')+'/' for rdir in self.meta.rundirs]
        
        return

class JWST(Pipeline):
    """
    JWST specifc Pipeline class.
    """
    def __init__(self, config_file):
        """
        Initialize a JWST specific Pipeline Class
        
        Note: this class only works with NIRCam so far.
        
        TODO: not all subarrays are assigned a PSF mask name from the CRDS yet.
              The assignments for all subarrays can be found at
              https://jwst-crds.stsci.edu/.
        
        Parameters
        ----------
        config_file : str
            Filepath to .yaml configuration file. 
        
        """
        
        # Intialise parent class
        super().__init__(config_file)

        # Get properties for JWST
        self.get_jwst_meta()

        return None

    def get_jwst_meta(self):
        """ 
        Define a range of parameters specific to JWST and its instruments
        """

        # Define telescope and instrument properties.
        self.meta.diam = 6.5 # m; primary mirror diameter
        self.meta.iwa = 1. # pix; inner working angle
        self.meta.owa = 150. # pix; outer working angle
        self.meta.pxsc_sw = 31.1 # mas; pixel scale of the short wavelength module
        self.meta.pxsc_lw = 63. # mas; pixel scale of the long wavelength module

        # Ancillary directories
        if not os.path.isdir(self.meta.ancildir):
            self.meta.ancildir = self.meta.odir + 'ANCILLARY/'
        self.meta.psfmaskdir = self.meta.ancildir  + 'psfmasks/'
        self.meta.offsetpsfdir = self.meta.ancildir  + 'offsetpsfs/'
    

        # Effective wavelength of the NIRCam filters from the SVO Filter
        # Profile Service.
        self.meta.wave = {'F182M': 1.838899e-6, # m
                     'F187N': 1.873722e-6, # m
                     'F200W': 1.968088e-6, # m
                     'F210M': 2.090846e-6, # m
                     'F212N': 2.121193e-6, # m
                     'F250M': 2.500588e-6, # m
                     'F300M': 2.981837e-6, # m
                     'F335M': 3.353823e-6, # m
                     'F356W': 3.528743e-6, # m
                     'F360M': 3.614837e-6, # m
                     'F410M': 4.072309e-6, # m
                     'F430M': 4.278486e-6, # m
                     'F444W': 4.350440e-6, # m
                     'F460M': 4.626991e-6, # m
                     'F480M': 4.813906e-6, # m
                     }
        
        # Filter zero point of the NIRCam filters from the SVO Filter Profile
        # Service.
        self.meta.F0 = {'F182M': 858.76, # Jy
                   'F187N': 813.41, # Jy
                   'F200W': 759.59, # Jy
                   'F210M': 701.37, # Jy
                   'F212N': 690.90, # Jy
                   'F250M': 515.84, # Jy
                   'F300M': 377.25, # Jy
                   'F335M': 305.60, # Jy
                   'F356W': 272.62, # Jy
                   'F360M': 266.13, # Jy
                   'F410M': 213.27, # Jy
                   'F430M': 195.51, # Jy
                   'F444W': 184.42, # Jy
                   'F460M': 168.30, # Jy
                   'F480M': 157.04, # Jy
                   }
        
        # PSF mask names from the CRDS.
        self.meta.psfmask = {'F250M_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0066',
                        'F300M_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0054',
                        'F356W_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0002',
                        'F410M_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0067',
                        'F444W_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0075',
                        'F356W_MASKA430R_SUB320A430R': 'jwst_nircam_psfmask_0065',
                        'F444W_MASKA430R_SUB320A430R': 'jwst_nircam_psfmask_0004',
                        'F250M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0042',
                        'F300M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0045',
                        'F335M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0003',
                        'F410M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0048',
                        'F430M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0055',
                        'F460M_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0058',
                        'narrow_MASKALWB_SUB320ALWB': 'jwst_nircam_psfmask_0042',
                        }
        
        # PSF position with respect to the NRCA4_MASKSWB and the
        # NRCA5_MASKLWB subarray, respectively, for each NIRCam filter from
        # pySIAF.
        self.meta.offset_swb = {'F182M': -1.743, # arcsec
                           'F187N': -1.544, # arcsec
                           'F210M': -0.034, # arcsec
                           'F212N': 0.144, # arcsec
                           'F200W': 0.196, # arcsec
                           'narrow': -8.053, # arcsec
                           }
        self.meta.offset_lwb = {'F250M': 6.565, # arcsec
                           'F300M': 5.042, # arcsec
                           'F277W': 4.917, # arcsec
                           'F335M': 3.875, # arcsec
                           'F360M': 3.057, # arcsec
                           'F356W': 2.327, # arcsec
                           'F410M': 1.622, # arcsec
                           'F430M': 0.998, # arcsec
                           'F460M': -0.094, # arcsec
                           'F444W': -0.723, # arcsec
                           'F480M': -0.854, # arcsec
                           'narrow': 8.302, # arcsec
                           }
        
        return

    def run_all(self, skip_ramp=False, skip_imgproc=False, skip_sub=False, skip_rawcon=False, skip_calcon=False, skip_comps=False):
        """
        Single function to run all pipeline stages in sequence based
        on the "do_X" flags within the config file. 
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
        self.meta.do_rampfit = True

        # Run ramp fitting
        ramp = rampfit.stsci_ramp_fitting(self.meta)

        return 

    def imgprocess(self):
        '''
        Wrapper function for image processing stage
        '''
        # Set meta flag to True if not already
        self.meta.do_imgprocess = True
        
        #Run image processing
        img = imgprocess.stsci_image_processing(self.meta)

        return

    def subtraction(self):
        '''
        Wrapper function for subtraction stage
        '''
        # Set meta flag to True if not already
        self.meta.do_subtraction = True

        #Run subtraction
        sub = subtraction.klip_subtraction(self.meta)
        return

    def raw_contrast(self):
        '''
        Wrapper function for raw contrast  stage
        '''
        # Set meta flag to True if not already
        self.meta.do_raw_contrast = True

        #Run raw contrast calculation
        raw_contrast = contrast.raw_contrast_curve(self.meta)
        return

    def cal_contrast(self):
        '''
        Wrapper function for calibrated contrast stage
        '''
        # Set meta flag to True if not already
        self.meta.do_cal_contrast = True
        # Run calibrated contrast calculation
        cal_contrast = contrast.calibrated_contrast_curve(self.meta)
        return

    def companions(self):
        '''
        Wrapper function for companion stage
        '''
        # Set meta flag to True if not already
        self.meta.do_companion = True
        #Run companion photometry / astrometry
        extract_comps = companion.extract_companions(self.meta)
        return