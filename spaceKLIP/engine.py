from __future__ import division
# =============================================================================
# IMPORTS
# =============================================================================

import glob, os, re
import json

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

from astroquery.svo_fps import SvoFps

import pysiaf

from . import io
from . import rampfit
from . import imgprocess
from . import subtraction
from . import contrast
from . import companion

# Define logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# =============================================================================
# MAIN
# =============================================================================

class Meta():
    """
    A meta class to hold information throughout the pipeline process,
    including user inputs.

    """

    def __init__(self):

        return None

    def save(self):
        """
        TODO: write function that could save all of the used parameters for
        debugging purposes.

        """

        return None

class Pipeline():
    """
    Generic pipeline class.

    """

    def __init__(self, config_file='template'):
        """
        Initialize a generic pipeline class by reading a config file and
        saving the inputs to a Meta object.

        Parameters
        ----------
        config_file : str
            File path of the YAML configuration file containing the pipeline
            setup parameters.

        """

        # Initialize the meta object.
        self.meta = Meta()

        # Read the configuration parameters.
        if (config_file == 'template'):
            tempstring = '/../tests/example_config.yaml'
            config_file = os.path.join(os.path.dirname(__file__)+tempstring)
        config = io.read_config(config_file)

        # Assign the configuration parameters to meta class attributes.
        for key in config:
            setattr(self.meta, key, config[key])

        # Assign run directories from output folder. These will be overwritten if subtraction if performed.
        if (self.meta.rundirs != None) or (len(self.meta.rundirs) == 0):
            if len(self.meta.rundirs) == 0:
                log.warning('No run directory(ies) specified, looping over all run directories in output directory. Are you sure you want to do this?')
                self.meta.rundirs = [i+'/' for i in glob.glob(self.meta.odir+'*run*')]
            else:
                self.meta.rundirs = [self.meta.odir+rdir.replace('/', '')+'/' for rdir in self.meta.rundirs]

        return None


class JWST(Pipeline):
    """
    JWST-specifc pipeline class.

    """

    def __init__(self, config_file):
        """
        Initialize a JWST-specific pipeline class.

        Parameters
        ----------
        config_file : str
            File path of the YAML configuration file containing the pipeline
            setup parameters.

        """

        # Initialize the parent pipeline class.
        super().__init__(config_file)

        # Assign flags to track which pipeline stages have been performed.
        self.meta.done_rampfit = False
        self.meta.done_imgprocess = False
        self.meta.done_subtraction = False
        self.meta.done_raw_contrast = False
        self.meta.done_cal_contrast = False
        self.meta.done_companion = False

        # Get the JWST-specific metadata.
        self.get_jwst_meta()

        return None

    def sort_files(self):
        """Sort files into subdirectories like filter_mask (e.g., F300M_MASK335R)"""

        idir = self.meta.idir[:-1] if self.meta.idir[-1]=='/' else self.meta.idir
        outdir = os.path.dirname(idir)

        if hasattr(self.meta, 'filter') and self.meta.filter.lower()!='none':
            filter = self.meta.filter
        else:
            filter = None
        if hasattr(self.meta, 'coron_mask') and self.meta.coron_mask.lower()!='none':
            coron_mask = self.meta.coron_mask
        else:
            coron_mask = None

        indir = None if self.meta.data_dir.lower()=='none' else self.meta.data_dir

        io.sort_data_files(self.meta.pid, self.meta.sci_obs, self.meta.ref_obs, outdir, 
                           indir=indir, expid_sci=self.meta.expid_sci, 
                           filter=filter, coron_mask=coron_mask)

    def meta_checks(self):
        """Check some consistencies in the meta file"""

        meta = self.meta

        # Check if outlier correction / cleaning requested
        if hasattr(meta, 'outlier_corr') and ((meta.outlier_corr is not None) or (meta.outlier_corr.lower() != 'none')):
            outlier_type = meta.outlier_corr
        else:
            outlier_type = None

        # Was cleaning requested on existing cal data?
        do_clean_only = do_clean = False
        if hasattr(meta, 'outlier_only') and meta.outlier_only:
            do_clean_only = True
            do_clean = True  # Must be set to True
            if outlier_type is None:
                log.warning('Meta: outlier_only=True but outlier_corr not specified.')
        if outlier_type is not None:
            do_clean = True

        if hasattr(meta, 'use_cleaned'):
            if meta.used_clean and not do_clean:
                log.warning('Meta: use_cleaned=True for KLIP subtraction, but no cleaning options specified.')
            if not meta.used_clean and do_clean:
                log.warning('Meta: Image cleaning will be performed, but use_cleaned=False for KLIP subtraction.')

    def get_jwst_meta(self):
        """
        Get the JWST-specific metadata.

        """

        # Define the telescope and instrument properties.
        self.meta.diam = 6.5 # m; primary mirror diameter
        self.meta.iwa = 1. # pix; inner working angle
        self.meta.owa = 250. # pix; outer working angle

        # Define the ancillary directories.
        if (not os.path.isdir(self.meta.ancildir)):
            self.meta.ancildir = self.meta.odir+'ANCILLARY/'
            if not os.path.exists(self.meta.ancildir):
                os.makedirs(self.meta.ancildir)
        self.meta.psfmaskdir = self.meta.ancildir+'psfmasks/' # no longer required
        self.meta.offsetpsfdir = self.meta.ancildir+'offsetpsfs/'

        # Get the mean wavelengths and zero points of the NIRCam and the MIRI
        # filters. All filters are saved into the same dictionary. This works 
        # as long as the NIRCam and the MIRI filter names are distinct.
        self.meta.wave = {}
        self.meta.weff = {}
        self.meta.F0 = {}
        if hasattr(self.meta, 'use_svo'):
            # From the SVO Filter Profile Service
            if self.meta.use_svo == True:
                filter_list = SvoFps.get_filter_list(facility='JWST', instrument='NIRCAM')
                for i in range(len(filter_list)):
                    name = filter_list['filterID'][i]
                    name = name[name.rfind('.')+1:]
                    self.meta.wave[name] = filter_list['WavelengthMean'][i]/1e4*1e-6 # m
                    self.meta.weff[name] = filter_list['WidthEff'][i]/1e4*1e-6 # m
                    self.meta.F0[name] = filter_list['ZeroPoint'][i] # Jy
                filter_list = SvoFps.get_filter_list(facility='JWST', instrument='MIRI')
                for i in range(len(filter_list)):
                    name = filter_list['filterID'][i]
                    name = name[name.rfind('.')+1:]
                    self.meta.wave[name] = filter_list['WavelengthMean'][i]/1e4*1e-6 # m
                    self.meta.weff[name] = filter_list['WidthEff'][i]/1e4*1e-6 # m
                    self.meta.F0[name] = filter_list['ZeroPoint'][i] # Jy
                del filter_list
        else:
            # From the file in the resources directory (more accurate than SVO)
            filt_info_str = '/../resources/PCEs/filter_info.json'
            filt_info_file = os.path.join(os.path.dirname(__file__) + filt_info_str)
            with open(filt_info_file, 'r') as f:
                filt_info = json.load(f)
                for filt in list(filt_info.keys()):
                    self.meta.wave[filt] = filt_info[filt]['WavelengthMean']/1e4*1e-6 # m
                    self.meta.weff[filt] = filt_info[filt]['WidthEff']/1e4*1e-6 # m
                    self.meta.F0[filt] = filt_info[filt]['ZeroPoint'] # Jy 

        # Get the PSF reference position with respect to the NRCA5_MASKLWB and
        # the NRCA4_MASKSWB subarrays, respectively, for each NIRCam filter,
        # from pySIAF.
        self.siaf = pysiaf.Siaf('NIRCAM')
        self.meta.offset_lwb = {filt: self.get_bar_offset_from_siaf(filt, channel='LW')
                                for filt in ['F250M', 'F277W', 'F300M', 'F335M', 'F356W', 'F360M', 'F410M', 'F430M', 'F444W', 'F460M', 'F480M', 'narrow']} # arcsec
        self.meta.offset_swb = {filt: self.get_bar_offset_from_siaf(filt, channel='SW')
                                for filt in ['F182M', 'F187N', 'F200W', 'F210M', 'F212N', 'narrow']} # arcsec
        del self.siaf



        return None

    def get_bar_offset_from_siaf(self, filt, channel='LW'):
        """
        Get the PSF reference position with respect to the NRCA5_MASKLWB and
        the NRCA4_MASKSWB subarrays, respectively, from pySIAF.

        Parameters
        ----------
        filt : str
            Name of the NIRCam filter.
        channel : str, optional
            Long wavelength (LW) or short wavelength (SW) channel. The default
            is 'LW'.

        Returns
        -------
        bar_offset : float
            Offset of the PSF reference position with respect to the
            NRCA5_MASKLWB and the NRCA4_MASKSWB subarrays, respectively, in
            arcseconds.

        """
        if (channel == 'SW'):
            refapername = 'NRCA4_MASKSWB'
            apername = 'NRCA4_MASKSWB_'+filt.upper()
        else: # otherwise default to LW channel
            refapername = 'NRCA5_MASKLWB'
            apername = 'NRCA5_MASKLWB_'+filt.upper()
        offset_arcsec = np.sqrt((self.siaf.apertures[refapername].V2Ref-self.siaf.apertures[apername].V2Ref)**2+(self.siaf.apertures[refapername].V3Ref-self.siaf.apertures[apername].V3Ref)**2)
        sign = np.sign(self.siaf.apertures[refapername].V2Ref-self.siaf.apertures[apername].V2Ref)

        return sign*offset_arcsec

    def run_all(self, skip_ramp=False, skip_imgproc=False, skip_sub=False,
                skip_rawcon=False, skip_calcon=False, skip_comps=False):
        """
        Single function to run all pipeline stages in sequence.

        Parameters
        ----------
        skip_ramp : bool
            Skip the ramp fitting (JWST stage 1) stage?
        skip_imgproc : bool
            Skip the image processing (JWST stage 2) stage?
        skip_sub : bool
            Skip the KLIP subtraction stage?
        skip_rawcon : bool
            Skip the raw contrast estimation stage?
        skip_calcon : bool
            Skip the calibrated contrast estimation stage?
        skip_comps : bool
            Skip the companion property extraction stage?

        """

        if (not skip_ramp):
            self.rampfit()
        if (not skip_imgproc):
            self.imgprocess()
        if (not skip_sub):
            self.subtraction()
        if (not skip_rawcon):
            self.raw_contrast()
        if (not skip_calcon):
            self.cal_contrast()
        if (not skip_comps):
            self.companions()

        return None

    def rampfit(self):
        """
        Wrapper function for the ramp fitting stage.

        """

        # Run ramp fitting stage.
        ramp = rampfit.stsci_ramp_fitting(self.meta)

        # Set the meta flag to True.
        self.meta.done_rampfit = True

        return None

    def imgprocess(self):
        """
        Wrapper function for the image processing stage.

        """

        # Run image processing stage.
        img = imgprocess.stsci_image_processing(self.meta)

        # Set the meta flag to True.
        self.meta.done_imgprocess = True

        return None

    def subtraction(self):
        """
        Wrapper function for the KLIP subtraction stage.

        """

        # Run KLIP subtraction stage.
        sub = subtraction.perform_subtraction(self.meta)

        # Set the meta flag to True.
        self.meta.done_subtraction = True

        return None

    def raw_contrast(self):
        """
        Wrapper function for the raw contrast estimation stage.

        """

        # Run raw contrast estimation stage.
        raw_contrast = contrast.raw_contrast_curve(self.meta)

        # Set the meta flag to True.
        self.meta.done_raw_contrast = True

        return None

    def cal_contrast(self):
        """
        Wrapper function for the calibrated contrast estimation stage.

        """

        # Run calibrated contrast estimation stage.
        cal_contrast = contrast.calibrated_contrast_curve(self.meta)

        # Set the meta flag to True.
        self.meta.done_cal_contrast = True

        return None

    def companions(self):
        """
        Wrapper function for the companion property extraction stage.

        """

        # Run companion property extraction stage.
        extract_comps = companion.extract_companions(self.meta)

        # Set the meta flag to True.
        self.meta.done_companion = True

        return None
