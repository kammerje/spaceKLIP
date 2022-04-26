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
from jwst import datamodels
from jwst.coron import AlignRefsStep

from . import io
from . import subtraction
from . import contrast
from . import companion
from . import utils

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

        # Create an astropy table for each unique set of observing parameters
        # (filter, coronagraph, ...). Save all information that is needed
        # later into this table. Finally, save all astropy tables into a
        # dictionary called meta.obs.
        ftype = 'calints' # only consider files in the input directory that contain this string
        fitsfiles_all = np.array([f for f in os.listdir(self.meta.idir) if ftype in f and f.endswith('.fits')])
        fitsfiles_use = []
        for i in range(len(fitsfiles_all)):
            if pyfits.getheader(self.meta.idir+fitsfiles_all[i])['EXP_TYPE'] in ['NRC_IMAGE', 'NRC_CORON', 'MIR_IMAGE', 'MIR_LYOT', 'MIR_4QPM']:
                fitsfiles_use += [fitsfiles_all[i]]
        fitsfiles_use = np.array(fitsfiles_use)
        Nfitsfiles = len(fitsfiles_use)

        nrc = webbpsf.NIRCam()
        mir = webbpsf.MIRI()

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
        APERNAME = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        PIXSCALE = np.empty(Nfitsfiles) # mas
        PA_V3 = np.empty(Nfitsfiles) # deg
        HASH = np.empty(Nfitsfiles, dtype=np.dtype('U100'))
        for i in range(Nfitsfiles):
            hdul = pyfits.open(self.meta.idir+fitsfiles_use[i])
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
            APERNAME[i] = head['APERNAME']
            if 'NIRCAM' in INSTRUME[i]:
                if 'LONG' in DETECTOR[i]:
                    PIXSCALE[i] = nrc._pixelscale_long*1e3 # mas
                else:
                    PIXSCALE[i] = nrc._pixelscale_short*1e3 # mas
            elif 'MIRI' in INSTRUME[i]:
                PIXSCALE[i] = mir.pixelscale*1e3 # mas
            else:
                raise UserWarning('Unknown instrument')
            head = hdul[1].header
            PA_V3[i] = head['ROLL_REF'] # deg (N over E)
            HASH[i] = INSTRUME[i]+'_'+DETECTOR[i]+'_'+FILTER[i]+'_'+PUPIL[i]+'_'+CORONMSK[i]+'_'+SUBARRAY[i]
            hdul.close()

        del nrc
        del mir

        # Data is grouped according to unique hash
        HASH_unique = np.unique(HASH)
        NHASH_unique = len(HASH_unique)
        self.meta.obs = {}
        for i in range(NHASH_unique):
            ww = HASH == HASH_unique[i]
            dpts = SUBPXPTS[ww]
            dpts_unique = np.unique(dpts)
            if len(dpts_unique) == 2 and dpts_unique[0] == 1:
                ww_sci = np.where(dpts == dpts_unique[0])[0]
                ww_cal = np.where(dpts == dpts_unique[1])[0]
            else:
                raise UserWarning('Science and reference PSFs are identified based on their number of dither positions, assuming that there is no dithering for the science PSFs')
            tab = Table(names=('TYP', 'TARGPROP', 'TARG_RA', 'TARG_DEC', 'READPATT', 'NINTS', 'NGROUPS', 'NFRAMES', 'EFFINTTM', 'APERNAME', 'PIXSCALE', 'PA_V3', 'FITSFILE'), dtype=('S', 'S', 'f', 'f', 'S', 'i', 'i', 'i', 'f', 'S', 'f', 'f', 'S'))
            for j in range(len(ww_sci)):
                tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], APERNAME[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], PA_V3[ww][ww_sci][j], self.meta.idir+fitsfiles_use[ww][ww_sci][j]))
            for j in range(len(ww_cal)):
                tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], APERNAME[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], PA_V3[ww][ww_cal][j], self.meta.idir+fitsfiles_use[ww][ww_cal][j]))
            self.meta.obs[HASH_unique[i]] = tab.copy()
            del tab

        if self.meta.verbose:
            print('--> Identified %.0f observation sequences' % len(self.meta.obs))
            for i, key in enumerate(self.meta.obs.keys()):
                print('--> Sequence %.0f: ' % (i+1)+key)
                print_table = copy.deepcopy(self.meta.obs[key])
                print_table.remove_column('FITSFILE')
                print_table.pprint(max_lines=100, max_width=1000)

        # Find the maximum numbasis based on the number of available
        # calibrator frames
        self.get_maxnumbasis()

        # Gather magnitudes for the target star
        self.meta.mstar = utils.get_stellar_magnitudes(self.meta)

        # Get properties for JWST
        self.get_jwst_meta()

        # Get the correct bar offset for each observing sequence.
        self.meta.bar_offset = {}
        for key in self.meta.obs.keys():
            temp = [s.start() for s in re.finditer('_', key)]
            filt = key[temp[1]+1:temp[2]].upper()
            if ('MASKALWB' in key.upper()):
                if ('NARROW' in self.meta.obs[key]['APERNAME'][0].upper()):
                    self.meta.bar_offset[key] = self.meta.offset_lwb['narrow']
                else:
                    self.meta.bar_offset[key] = self.meta.offset_lwb[filt]
            elif ('MASKASWB' in key.upper()):
                if ('NARROW' in self.meta.obs[key]['APERNAME'][0].upper()):
                    self.meta.bar_offset[key] = self.meta.offset_swb['narrow']
                else:
                    self.meta.bar_offset[key] = self.meta.offset_swb[filt]
            else:
                self.meta.bar_offset[key] = None

        return None

    def get_maxnumbasis(self):
        """
        Find the maximum numbasis based on the number of available calibrator
        frames.
        """
        
        # The number of available calibrator frames can be found in the
        # self.meta.obs table
        self.meta.maxnumbasis = {}
        for i, key in enumerate(self.meta.obs.keys()):
            ww = self.meta.obs[key]['TYP'] == 'CAL'
            self.meta.maxnumbasis[key] = np.sum(self.meta.obs[key]['NINTS'][ww], dtype=int)
        
        return

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

        # PSF mask names from the CRDS
        step = AlignRefsStep()
        self.meta.psfmask = {}
        for key in self.meta.obs.keys():
            model = datamodels.open(self.meta.obs[key]['FITSFILE'][0])            
            self.meta.psfmask[key] = step.get_reference_file(model, 'psfmask')
        del step

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

    def run(self):
        """
        Run reduction based on inputs from the config file.
        """

        if self.meta.do_subtraction:
            sub = subtraction.klip_subtraction(self.meta)
        if self.meta.do_raw_contrast:
            raw_contrast = contrast.raw_contrast_curve(self.meta)
        if self.meta.do_cal_contrast:
            cal_contrast = contrast.calibrated_contrast_curve(self.meta)
        if self.meta.do_companion:
            extract_comps = companion.extract_companions(self.meta)
        return
