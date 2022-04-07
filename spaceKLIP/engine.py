import os
import numpy as np
import astropy.io.fits as pyfits
from astropy.table import Table

from . import io
from . import subtraction
from . import contrast
from . import companion
from . import utils

class Params():
    """
    Class to hold a variety of parameters for the reduction stages.
    """
    def __init__(self):
        return

    def save(self):
        # TODO, write function that could save all of the used parameters
        # for debugging purposes.
        return

class Reduction():
    """
    Generic Reduction Class
    """
    def __init__(self, config_file):
        """
        Initialise a generic reduction class by reading a config file and 
        assigning parameters to a Params object:
        """

        # Initialise Params object
        self.params = Params()

        # Read in configuration parameters
        config = io.read_config(config_file)
        # Assign config parameters to class attributes
        for key in config:
            setattr(self.params, key, config[key])

        return

class JWSTReduction(Reduction):
    """
    JWST Reduction specific class.
    """
    def __init__(self, config_file):
        """
        Initialize a JWST Reduction Class
        
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
        self.get_jwst_params()

        # Create an astropy table for each unique set of observing parameters
        # (filter, coronagraph, ...). Save all information that is needed
        # later into this table. Finally, save all astropy tables into a
        # dictionary called self.obs.
        ftyp = 'calints' # only consider files in the input directory that contain this string
        fitsfiles = np.array([f for f in os.listdir(self.params.idir) if ftyp in f and f.endswith('.fits')])
        Nfitsfiles = len(fitsfiles)
        TARGPROP = []
        TARG_RA = [] # deg
        TARG_DEC = [] # deg
        INSTRUME = []
        DETECTOR = []
        FILTER = []
        PUPIL = []
        CORONMSK = []
        READPATT = []
        NINTS = []
        NGROUPS = []
        NFRAMES = []
        EFFINTTM = [] # s
        SUBARRAY = []
        SUBPXPTS = []
        PIXSCALE = [] # mas
        PA_V3 = [] # deg
        HASH = []
        for i in range(Nfitsfiles):
            hdul = pyfits.open(self.params.idir+fitsfiles[i])
            head = hdul[0].header
            TARGPROP += [str(head['TARGPROP'])]
            TARG_RA += [float(head['TARG_RA'])] # deg
            TARG_DEC += [float(head['TARG_DEC'])] # deg
            INSTRUME += [str(head['INSTRUME'])]
            DETECTOR += [str(head['DETECTOR'])]
            FILTER += [str(head['FILTER'])]
            PUPIL += [str(head['PUPIL'])]
            CORONMSK += [str(head['CORONMSK'])]
            READPATT += [str(head['READPATT'])]
            NINTS += [int(head['NINTS'])]
            NGROUPS += [int(head['NGROUPS'])]
            NFRAMES += [int(head['NFRAMES'])]
            EFFINTTM += [float(head['EFFINTTM'])] # s
            SUBARRAY += [str(head['SUBARRAY'])]
            try:
                SUBPXPTS += [int(head['SUBPXPTS'])]
            except:
                SUBPXPTS += [1]
            if ('LONG' in DETECTOR[-1]):
                PIXSCALE += [self.params.pxsc_lw] # mas
            else:
                PIXSCALE += [self.params.pxsc_sw] # mas
            head = hdul[1].header
            PA_V3 += [float(head['PA_V3'])] # deg
            HASH += [INSTRUME[-1]+'_'+DETECTOR[-1]+'_'+FILTER[-1]+'_'+PUPIL[-1]+'_'+CORONMSK[-1]+'_'+SUBARRAY[-1]]
            hdul.close()
        TARGPROP = np.array(TARGPROP)
        TARG_RA = np.array(TARG_RA) # deg
        TARG_DEC = np.array(TARG_DEC) # deg
        INSTRUME = np.array(INSTRUME)
        DETECTOR = np.array(DETECTOR)
        FILTER = np.array(FILTER)
        PUPIL = np.array(PUPIL)
        CORONMSK = np.array(CORONMSK)
        READPATT = np.array(READPATT)
        NINTS = np.array(NINTS)
        NGROUPS = np.array(NGROUPS)
        NFRAMES = np.array(NFRAMES)
        EFFINTTM = np.array(EFFINTTM) # s
        SUBARRAY = np.array(SUBARRAY)
        SUBPXPTS = np.array(SUBPXPTS)
        PIXSCALE = np.array(PIXSCALE) # mas
        PA_V3 = np.array(PA_V3) # deg
        HASH = np.array(HASH)
        HASH_unique = np.unique(HASH)
        NHASH_unique = len(HASH_unique)
        self.obs = {}
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
                tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], PA_V3[ww][ww_sci][j], self.params.idir+fitsfiles[ww][ww_sci][j]))
            for j in range(len(ww_cal)):
                tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], PA_V3[ww][ww_cal][j], self.params.idir+fitsfiles[ww][ww_cal][j]))
            self.obs[HASH_unique[i]] = tab.copy()
        
        if (self.params.verbose == True):
            print('--> Identified %.0f observation sequences' % len(self.obs))
            for i, key in enumerate(self.obs.keys()):
                print('--> Sequence %.0f: ' % (i+1)+key)
                print(self.obs[key])
        
        # Find the maximum numbasis based on the number of available
        # calibrator frames.
        self.get_maxnumbasis()

        # Gather magnitudes for the target star
        self.params.mstar = utils.get_stellar_magnitudes(self.params, self.obs)
        
        return None

    def get_maxnumbasis(self):
        """
        Find the maximum numbasis based on the number of available calibrator
        frames.
        """
        
        # The number of available calibrator frames can be found in the
        # self.obs table.
        self.params.maxnumbasis = {}
        for i, key in enumerate(self.obs.keys()):
            ww = self.obs[key]['TYP'] == 'CAL'
            self.params.maxnumbasis[key] = np.sum(self.obs[key]['NINTS'][ww])
        
        return

    def get_jwst_params(self):
        """ 
        Define a range of parameters specific to JWST and its instruments
        """

        # Define telescope and instrument properties.
        self.params.diam = 6.5 # m; primary mirror diameter
        self.params.iwa = 1. # pix; inner working angle
        self.params.owa = 150. # pix; outer working angle
        self.params.pxsc_sw = 31.1 # mas; pixel scale of the short wavelength module
        self.params.pxsc_lw = 63. # mas; pixel scale of the long wavelength module
        self.params.gain_sw = 2.01 # e-/DN; gain of the short wavelength module
        self.params.gain_lw = 1.83 # e-/DN; gain of the long wavelength module
        

        # Effective wavelength of the NIRCam filters from the SVO Filter
        # Profile Service.
        self.params.wave = {'F182M': 1.838899e-6, # m
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
        self.params.F0 = {'F182M': 858.76, # Jy
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
        self.params.psfmask = {'F250M_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0066',
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
        self.params.offset_swb = {'F182M': -1.743, # arcsec
                           'F187N': -1.544, # arcsec
                           'F210M': -0.034, # arcsec
                           'F212N': 0.144, # arcsec
                           'F200W': 0.196, # arcsec
                           'narrow': -8.053, # arcsec
                           }
        self.params.offset_lwb = {'F250M': 6.565, # arcsec
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

    def run(self):
        """
        Run reduction based on inputs from the config file. 
        """
        sub = subtraction.klip_subtraction(self.params, self.obs)
        raw_contrast = contrast.raw_contrast_curve(self.params, self.obs)
        cal_contrast = contrast.calibrated_contrast_curve(self.params, self.obs)
        extract_comps = companion.extract_companions(self.params, self.obs)

        return
    
