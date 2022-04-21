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

        # If we have already done Stage 1 and 2 of the pipeline files, get the observations.
        if (self.meta.do_imgprocess != True) and (self.meta.do_rampfit != True):
            self.extract_obs(self.meta.idir)
            # Find the maximum numbasis based on the number of available
            # calibrator frames.
            self.get_maxnumbasis()
            # Gather magnitudes for the target star
            self.meta.mstar = utils.get_stellar_magnitudes(self.meta)

        return None

    def get_maxnumbasis(self):
        """
        Find the maximum numbasis based on the number of available calibrator
        frames.
        """
        
        # The number of available calibrator frames can be found in the
        # self.obs table.
        self.meta.maxnumbasis = {}
        for i, key in enumerate(self.meta.obs.keys()):
            ww = self.meta.obs[key]['TYP'] == 'CAL'
            self.meta.maxnumbasis[key] = np.sum(self.meta.obs[key]['NINTS'][ww], dtype=int)
        
        return

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

    def extract_obs(self, idir):
        # Create an astropy table for each unique set of observing parameters
        # (filter, coronagraph, ...). Save all information that is needed
        # later into this table. Finally, save all astropy tables into a
        # dictionary called meta.obs.
        ftyp = 'calints' # only consider files in the input directory that contain this string
        fitsfiles = np.array([f for f in os.listdir(idir) if ftyp in f and f.endswith('.fits')])
        Nfitsfiles = len(fitsfiles)
        
        TARGPROP = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        TARG_RA = np.empty(Nfitsfiles) # deg
        TARG_DEC = np.empty(Nfitsfiles) # deg
        INSTRUME = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        DETECTOR = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        FILTER = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        PUPIL = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        CORONMSK = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        READPATT = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        NINTS = np.empty(Nfitsfiles, dtype=int)
        NGROUPS = np.empty(Nfitsfiles, dtype=int)
        NFRAMES = np.empty(Nfitsfiles, dtype=int)
        EFFINTTM = np.empty(Nfitsfiles) # s
        SUBARRAY = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        SUBPXPTS = np.empty(Nfitsfiles, dtype=int)
        PIXSCALE = np.empty(Nfitsfiles) # mas
        PA_V3 = np.empty(Nfitsfiles) # deg
        HASH = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        for i in range(Nfitsfiles):
            hdul = pyfits.open(idir+fitsfiles[i])
            head = hdul[0].header

            TARGPROP[i] = head['TARGPROP']
            TARG_RA[i] = head['TARG_RA'] # deg
            TARG_DEC[i] = head['TARG_DEC'] # deg
            INSTRUME[i] = head['INSTRUME']
            DETECTOR[i] = head['DETECTOR']
            FILTER[i] = head['FILTER']
            PUPIL[i] = head['PUPIL']
            CORONMSK[i] = head['CORONMSK']
            READPATT[i] = head['READPATT']
            NINTS[i] = head['NINTS']
            NGROUPS[i] = head['NGROUPS']
            NFRAMES[i] = head['NFRAMES']
            EFFINTTM[i] = head['EFFINTTM'] # s
            SUBARRAY[i] = head['SUBARRAY']
            try:
                SUBPXPTS[i] = head['SUBPXPTS']
            except:
                SUBPXPTS[i] = 1
            if ('LONG' in DETECTOR[i]):
                PIXSCALE[i] = self.meta.pxsc_lw # mas
            else:
                PIXSCALE[i] = self.meta.pxsc_sw # mas
            head = hdul[1].header
            PA_V3[i] = head['PA_V3'] # deg
            HASH[i] = INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]
            hdul.close()

        HASH_unique = np.unique(HASH)
        NHASH_unique = len(HASH_unique)
        self.meta.obs = {}
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
                tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], PA_V3[ww][ww_sci][j], idir+fitsfiles[ww][ww_sci][j]))
            for j in range(len(ww_cal)):
                tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], PA_V3[ww][ww_cal][j], idir+fitsfiles[ww][ww_cal][j]))
            self.meta.obs[HASH_unique[i]] = tab.copy()
        
        if (self.meta.verbose == True):
            print('--> Identified %.0f observation sequences' % len(self.meta.obs))
            for i, key in enumerate(self.meta.obs.keys()):
                print('--> Sequence %.0f: ' % (i+1)+key)
                print_table = copy.deepcopy(self.meta.obs[key])
                print_table.remove_column('FITSFILE')
                print_table.pprint(max_lines=100, max_width=1000)

        return

    def run(self):
        """
        Run reduction based on inputs from the config file. 
        """
        if self.meta.do_rampfit:
            ramp = rampfit.stsci_ramp_fitting(self.meta)
        if self.meta.do_imgprocess:
            img = imgprocess.stsci_image_processing(self.meta)
            self.extract_obs(self.meta.odir+'IMGPROCESS/')
            # Find the maximum numbasis based on the number of available
            # calibrator frames.
            self.get_maxnumbasis()
            # Gather magnitudes for the target star
            self.meta.mstar = utils.get_stellar_magnitudes(self.meta)
        if self.meta.do_subtraction:
            sub = subtraction.klip_subtraction(self.meta)
        if self.meta.do_raw_contrast:
            raw_contrast = contrast.raw_contrast_curve(self.meta)
        if self.meta.do_cal_contrast:
            cal_contrast = contrast.calibrated_contrast_curve(self.meta)
        if self.meta.do_companion:
            extract_comps = companion.extract_companions(self.meta)
        return
    
