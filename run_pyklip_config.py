from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})

import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import csv
import os
import re
import sys
import urllib
import yaml

from astropy.table import Table
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import rotate, shift
from scipy.optimize import leastsq, least_squares

import webbpsf

import pyklip.fakes as fakes
import pyklip.fitpsf as fitpsf
import pyklip.fm as fm
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.instruments.JWST as JWST
import pyklip.klip as klip
import pyklip.parallelized as parallelize

# Only required by old transmission model.
# sys.path.append('/Users/jkammerer/Documents/Code/opticstools/opticstools')
# import opticstools as ot

nircam = webbpsf.NIRCam()

rad2mas = 180./np.pi*3600.*1000.
mas2rad = np.pi/180./3600./1000.

with open("config.yaml", "r") as stream:
    try:
        config = yaml.safe_load(stream)
    except:
        raise yaml.YAMLError

# =============================================================================
# PARAMETERS
# =============================================================================
idir = config['idir'] # input directory with stage 2 calibrated data
odir = config['odir'] # output directory for the plots and pyKLIP products
transmissiondir = config['transmissiondir'] # directory containing the transmission functions from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-coronagraphic-occulting-masks-and-lyot-stops
psfmaskdir = config['psfmaskdir'] # directory for the PSF masks (will be downloaded automatically from CRDS)
use_psfmask = config['use_psfmask'] # if true use PSF masks from CRDS, if false use transmission functions
offsetpsfdir = config['offsetpsfdir'] # directory for the offset PSFs (will be generated automatically using WebbPSF)

mode = config['mode'] # list of modes for pyKLIP, will loop through all
annuli = config['annuli'] # list of number of annuli for pyKLIP, will loop through all
subsections = config['subsections'] # list of number of annuli for pyKLIP, will loop through all
numbasis = config['numbasis'] # list of number of basis vectors for pyKLIP, will loop through all
verbose = config['verbose'] # if true print status updates

fiducial_point_override = config['fiducial_point_override'] # if true uses narrow end of the bar masks, if false use filter dependent position

# Host star magnitude in each filter. Must contain one entry for each filter
# used in the data in the input directory.
mstar = {'F250M': 6, # vegamag
         'F300M': 6, # vegamag
         'F356W': 6, # vegamag
         'F410M': 6, # vegamag
         'F444W': 6, # vegamag
         }

ra_off = config['ra_off'] # mas; RA offset of the known companions in the same order as in the NIRCCoS config file
de_off = config['de_off'] # mas; DEC offset of the known companions in the same order as in the NIRCCoS config file
pa_ranges_bar = config['pa_ranges_bar']# deg; list of tuples defining the pizza slices that shall be considered when computing the contrast curves for the bar masks

seps_inject_rnd = config['seps_inject_rnd'] # pix; list of separations at which fake planets shall be injected to compute the calibrated contrast curve for the round masks
pas_inject_rnd = config['pas_inject_rnd'] # deg; list of position angles at which fake planets shall be injected to compute the calibrated contrast curve for the round masks
seps_inject_bar = config['seps_inject_bar'] # pix; list of separations at which fake planets shall be injected to compute the calibrated contrast curve for the bar masks
pas_inject_bar = config['pas_inject_bar'] # deg; list of position angles at which fake planets shall be injected to compute the calibrated contrast curve for the bar masks
KL = config['KL'] # index of the KL component for which the calibrated contrast curve and the companion properties shall be computed


# =============================================================================
# PROCESSOR
# =============================================================================

class processor():
    
    def __init__(self,
                 idir,
                 odir,
                 mode=['RDI'],
                 annuli=[1],
                 subsections=[1],
                 numbasis=[1, 2, 5, 10, 20, 50, 100],
                 verbose=True):
        """
        Initialize the pyKLIP processor class.
        
        Note: this class only works with NIRCam so far.
        
        TODO: not all subarrays are assigned a PSF mask name from the CRDS yet.
              The assignments for all subarrays can be found at
              https://jwst-crds.stsci.edu/.
        
        Parameters
        ----------
        idir: str
            Input directory with stage 2 calibrated data.
        odir: str
            Output directory for the plots and pyKLIP products.
        mode: list of str
            List of modes for pyKLIP, will loop through all.
        annuli: list of int
            List of number of annuli for pyKLIP, will loop through all.
        subsections: list of int
            List of number of subsections for pyKLIP, will loop through all.
        numbasis: list of int
            List of number of basis vectors for pyKLIP, will loop through all.
        verbose: bool
            If true print status updates.
        """
        
        # Define telescope and instrument properties.
        self.diam = 6.5 # m; primary mirror diameter
        self.iwa = 1. # pix; inner working angle
        self.owa = 150. # pix; outer working angle
        self.pxsc_sw = 31.1 # mas; pixel scale of the short wavelength module
        self.pxsc_lw = 63. # mas; pixel scale of the long wavelength module
        self.gain_sw = 2.01 # e-/DN; gain of the short wavelength module
        self.gain_lw = 1.83 # e-/DN; gain of the long wavelength module
        
        # Effective wavelength of the NIRCam filters from the SVO Filter
        # Profile Service.
        self.wave = {'F182M': 1.838899e-6, # m
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
        self.F0 = {'F182M': 858.76, # Jy
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
        self.psfmask = {'F250M_MASKA335R_SUB320A335R': 'jwst_nircam_psfmask_0066',
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
        self.offset_swb = {'F182M': -1.743, # arcsec
                           'F187N': -1.544, # arcsec
                           'F210M': -0.034, # arcsec
                           'F212N': 0.144, # arcsec
                           'F200W': 0.196, # arcsec
                           'narrow': -8.053, # arcsec
                           }
        self.offset_lwb = {'F250M': 6.565, # arcsec
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
        
        # Make inputs self.
        self.idir = idir
        self.odir = odir
        self.mode = mode
        self.annuli = annuli
        self.subsections = subsections
        self.numbasis = numbasis
        self.verbose = verbose
        
        # Create an astropy table for each unique set of observing parameters
        # (filter, coronagraph, ...). Save all information that is needed
        # later into this table. Finally, save all astropy tables into a
        # dictionary called self.obs.
        ftyp = 'calints' # only consider files in the input directory that contain this string
        fitsfiles = np.array([f for f in os.listdir(self.idir) if ftyp in f and f.endswith('.fits')])
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
            hdul = pyfits.open(idir+fitsfiles[i])
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
                PIXSCALE += [self.pxsc_lw] # mas
            else:
                PIXSCALE += [self.pxsc_sw] # mas
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
                tab.add_row(('SCI', TARGPROP[ww][ww_sci][j], TARG_RA[ww][ww_sci][j], TARG_DEC[ww][ww_sci][j], READPATT[ww][ww_sci][j], NINTS[ww][ww_sci][j], NGROUPS[ww][ww_sci][j], NFRAMES[ww][ww_sci][j], EFFINTTM[ww][ww_sci][j], PIXSCALE[ww][ww_sci][j], PA_V3[ww][ww_sci][j], idir+fitsfiles[ww][ww_sci][j]))
            for j in range(len(ww_cal)):
                tab.add_row(('CAL', TARGPROP[ww][ww_cal][j], TARG_RA[ww][ww_cal][j], TARG_DEC[ww][ww_cal][j], READPATT[ww][ww_cal][j], NINTS[ww][ww_cal][j], NGROUPS[ww][ww_cal][j], NFRAMES[ww][ww_cal][j], EFFINTTM[ww][ww_cal][j], PIXSCALE[ww][ww_cal][j], PA_V3[ww][ww_cal][j], idir+fitsfiles[ww][ww_cal][j]))
            self.obs[HASH_unique[i]] = tab.copy()
        
        if (self.verbose == True):
            print('--> Identified %.0f observation sequences' % len(self.obs))
            for i, key in enumerate(self.obs.keys()):
                print('--> Sequence %.0f: ' % (i+1)+key)
                print(self.obs[key])
        
        # Find the maximum numbasis based on the number of available
        # calibrator frames.
        self.get_maxnumbasis()
        
        return None
    
    def get_maxnumbasis(self):
        """
        Find the maximum numbasis based on the number of available calibrator
        frames.
        """
        
        # The number of available calibrator frames can be found in the
        # self.obs table.
        self.maxnumbasis = {}
        for i, key in enumerate(self.obs.keys()):
            ww = self.obs[key]['TYP'] == 'CAL'
            self.maxnumbasis[key] = np.sum(self.obs[key]['NINTS'][ww])
        
        return None
    
    def run_pyklip(self):
        """
        Run pyKLIP.
        """
        
        if (self.verbose == True):
            print('--> Running pyKLIP...')
        
        # Loop through all modes, numbers of annuli, and numbers of
        # subsections.
        Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
        counter = 1
        self.truenumbasis = {}
        for mode in self.mode:
            for annuli in self.annuli:
                for subsections in self.subsections:
                    
                    if (self.verbose == True):
                        sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                        sys.stdout.flush()
                    
                    # Create an output directory for each set of pyKLIP
                    # parameters.
                    odir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                    if (not os.path.exists(odir)):
                        os.makedirs(odir)
                    
                    # Loop through all sets of observing parameters. Only run
                    # pyKLIP if the corresponding KLmodes-all fits file does
                    # not exist yet.
                    for i, key in enumerate(self.obs.keys()):
                        self.truenumbasis[key] = [num for num in self.numbasis if (num <= self.maxnumbasis[key])]
                        if (os.path.exists(odir+key+'-KLmodes-all.fits')):
                            continue
                        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
                        filepaths = np.array(self.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                        ww_cal = np.where(self.obs[key]['TYP'] == 'CAL')[0]
                        psflib_filepaths = np.array(self.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                        dataset = JWST.JWSTData(filepaths=filepaths,
                                                psflib_filepaths=psflib_filepaths)
                        parallelize.klip_dataset(dataset=dataset,
                                                  mode=mode,
                                                  outputdir=odir,
                                                  fileprefix=key,
                                                  annuli=annuli,
                                                  subsections=subsections,
                                                  movement=1,
                                                  numbasis=self.truenumbasis[key],
                                                  calibrate_flux=False,
                                                  maxnumbasis=self.maxnumbasis[key],
                                                  psf_library=dataset.psflib,
                                                  highpass=False,
                                                  verbose=False)
                    counter += 1
        
        if (self.verbose == True):
            print('')
        
        return None
    
    def raw_contrast_curve(self,
                           mstar={}, # vegamag
                           ra_off=[], # mas
                           de_off=[], # mas
                           pa_ranges_bar=[]): # deg
        """
        Compute the raw contrast curves. Known companions and the location of
        the bar mask in both rolls can be masked out.
        
        Note: currently masks a circle with a radius of 12 FWHM = ~12 lambda/D
              around known companions. This was found to be a sufficient but
              necessary value based on tests with simulated data.
        
        TODO: tries to convert MJy/sr to contrast using the host star
              magnitude, but the results are wrong by a factor of ~5.
        
        TODO: currently uses WebbPSF to compute a theoretical offset PSF. It
              should be possible to use PSF stamps extracted from the
              astrometric confirmation images to determine the pupil mask
              throughput.
        
        Parameters
        ----------
        mstar: dict of float
            Host star magnitude in each filter. Must contain one entry for
            each filter used in the data in the input directory.
        ra_off: list of float
            RA offset of the known companions.
        de_off: list of float
            DEC offset of the known companions.
        pa_ranges_bar: list of tuple of float
            List of tuples defining the pizza slices that shall be considered
            when computing the contrast curves for the bar masks.
        """
        
        if (self.verbose == True):
            print('--> Computing raw contrast curve...')
        
        # Loop through all modes, numbers of annuli, and numbers of
        # subsections.
        Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
        counter = 1
        for mode in self.mode:
            for annuli in self.annuli:
                for subsections in self.subsections:
                    
                    if (self.verbose == True):
                        sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                        sys.stdout.flush()
                    
                    # Define the input and output directories for each set of
                    # pyKLIP parameters.
                    idir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                    odir = self.odir+mode+'_annu%.0f_subs%.0f/CONS/' % (annuli, subsections)
                    if (not os.path.exists(odir)):
                        os.makedirs(odir)
                    
                    # Loop through all sets of observing parameters.
                    for i, key in enumerate(self.obs.keys()):
                        hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                        data = hdul[0].data
                        pxsc = self.obs[key]['PIXSCALE'][0] # mas
                        cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                        temp = [s.start() for s in re.finditer('_', key)]
                        filt = key[temp[1]+1:temp[2]]
                        mask = key[temp[3]+1:temp[4]]
                        subarr = key[temp[4]+1:]
                        wave = self.wave[filt] # m
                        fwhm = wave/self.diam*rad2mas/pxsc # pix
                        hdul.close()
                        
                        # Mask out known companions and the location of the
                        # bar mask in both rolls.
                        data_masked = self.mask_companions(data, pxsc, cent, 12.*fwhm, ra_off, de_off)
                        if (mask in ['MASKASWB', 'MASKALWB']):
                            data_masked = self.mask_bar(data_masked, cent, pa_ranges_bar)
                        
                        # Plot.
                        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
                        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
                        f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                        ax[0].imshow(np.log10(np.abs(data[-1])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                        for i in range(len(ra_off)):
                            cc = plt.Circle((ra_off[i]/1000., de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[0].add_artist(cc)
                        ax[0].set_xlabel('$\Delta$RA [arcsec]')
                        ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                        ax[0].set_title('KLIP-subtracted')
                        ax[1].imshow(np.log10(np.abs(data_masked[-1])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                        ax[1].set_xlabel('$\Delta$RA [arcsec]')
                        ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                        if (mask in ['MASKASWB', 'MASKALWB']):
                            ax[1].set_title('Companions & bar masked')
                        else:
                            ax[1].set_title('Companions masked')
                        plt.tight_layout()
                        plt.savefig(odir+key+'-mask.pdf')
                        plt.close()
                        
                        # Convert the units and compute the contrast. Use an
                        # unocculted offset PSF from WebbPSF to obtain the
                        # peak count of the host star. The unocculted offset
                        # PSF from WebbPSF is normalized to an integrated
                        # source of 1 and takes into account the throughput
                        # of the pupil mask.
                        offsetpsf = self.get_offsetpsf(filt, mask, key)
                        Fstar = self.F0[filt]/10.**(mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                        Fdata = data_masked*pxsc**2/(180./np.pi*3600.*1000.)**2 # MJy; convert the data from MJy/sr to MJy
                        seps = [] # arcsec
                        cons = []
                        for i in range(Fdata.shape[0]):
                            sep, con = klip.meas_contrast(dat=Fdata[i]/Fstar, iwa=self.iwa, owa=self.owa, resolution=2.*fwhm, center=cent, low_pass_filter=False)
                            seps += [sep*pxsc/1000.] # arcsec
                            cons += [con]
                        seps = np.array(seps) # arcsec
                        cons = np.array(cons)
                        np.save(odir+key+'-seps.npy', seps) # arcsec
                        np.save(odir+key+'-cons.npy', cons)
                        
                        # Plot.
                        plt.figure(figsize=(6.4, 4.8))
                        ax = plt.gca()
                        for i in range(Fdata.shape[0]):
                            ax.plot(seps[i], cons[i], label=str(self.truenumbasis[key][i])+' KL')
                        ax.set_yscale('log')
                        ax.set_xlim([0., np.max(seps)]) # arcsec
                        ax.grid(axis='y')
                        ax.set_xlabel('Separation [arcsec]')
                        ax.set_ylabel('Contrast [5$\sigma$]')
                        ax.set_title('Raw contrast curve')
                        ax.legend(loc='upper right')
                        plt.tight_layout()
                        plt.savefig(odir+key+'-cons_raw.pdf')
                        plt.close()
                    counter += 1
        
        if (self.verbose == True):
            print('')
        
        return None
    
    def cal_contrast_curve(self,
                           mstar, # vegamag
                           ra_off=[], # mas
                           de_off=[], # mas
                           seps_inject_rnd=[], # pix
                           pas_inject_rnd=[], # deg
                           seps_inject_bar=[], # pix
                           pas_inject_bar=[], # deg
                           KL=-1,
                           overwrite=False):
        """
        Compute the calibrated contrast curves. Injection and recovery tests
        are performed to estimate the algo & coronmsk throughput.
        
        Note: do not inject fake planets on top of the bar mask in either of
              the rolls!
        
        Note: currently forces any fake companions to be injected at least 10
              FWHM = ~10 lambda/D away from other fake companions or known
              companions. This was found to be a sufficient but necessary
              value based on tests with simulated data.
        
        TODO: use a position dependent offset PSF from pyNRC instead of the
              completely unocculted offset PSF from WebbPSF.
        
        Parameters
        ----------
        mstar: dict of float
            Host star magnitude in each filter. Must contain one entry for
            each filter used in the data in the input directory.
        ra_off: list of float
            RA offset of the known companions.
        de_off: list of float
            DEC offset of the known companions.
        seps_inject_rnd: list of float
            List of separations at which fake planets shall be injected to
            compute the calibrated contrast curve for the round masks.
        pas_inject_rnd: list of float
            List of position angles at which fake planets shall be injected to
            compute the calibrated contrast curve for the round masks.
        seps_inject_bar: list of float
            List of separations at which fake planets shall be injected to
            compute the calibrated contrast curve for the bar masks.
        pas_inject_bar: list of float
            List of position angles at which fake planets shall be injected to
            compute the calibrated contrast curve for the bar masks.
        KL: int
            Index of the KL component for which the calibrated contrast curve
            and the companion properties shall be computed.
        overwrite: bool
            If true overwrite existing data.
        """
        
        if (self.verbose == True):
            print('--> Computing calibrated contrast curve...')
        
        # Make inputs arrays.
        seps_inject_rnd = np.array(seps_inject_rnd)
        pas_inject_rnd = np.array(pas_inject_rnd)
        seps_inject_bar = np.array(seps_inject_bar)
        pas_inject_bar = np.array(pas_inject_bar)
        
        # Loop through all modes, numbers of annuli, and numbers of
        # subsections.
        Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
        counter = 1
        for mode in self.mode:
            for annuli in self.annuli:
                for subsections in self.subsections:
                    
                    if (self.verbose == True):
                        sys.stdout.write('\r--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))
                        sys.stdout.flush()
                    
                    # Define the input and output directories for each set of
                    # pyKLIP parameters.
                    idir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                    odir = self.odir+mode+'_annu%.0f_subs%.0f/CONS/' % (annuli, subsections)
                    if (not os.path.exists(odir)):
                        os.makedirs(odir)
                    
                    # Loop through all sets of observing parameters.
                    for i, key in enumerate(self.obs.keys()):
                        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
                        filepaths = np.array(self.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                        ww_cal = np.where(self.obs[key]['TYP'] == 'CAL')[0]
                        psflib_filepaths = np.array(self.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                        hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                        data = hdul[0].data
                        pxsc = self.obs[key]['PIXSCALE'][0] # mas
                        cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                        temp = [s.start() for s in re.finditer('_', key)]
                        filt = key[temp[1]+1:temp[2]]
                        mask = key[temp[3]+1:temp[4]]
                        subarr = key[temp[4]+1:]
                        wave = self.wave[filt] # m
                        fwhm = wave/self.diam*rad2mas/pxsc # pix
                        hdul.close()
                        
                        # Load raw contrast curves. If overwrite is false,
                        # check whether the calibrated contrast curves have
                        # been computed already.
                        seps = np.load(odir+key+'-seps.npy')[KL] # arcsec
                        cons = np.load(odir+key+'-cons.npy')[KL]
                        if (overwrite == False):
                            try:
                                flux_all = np.load(odir+key+'-flux_all.npy') # MJy/sr
                                seps_all = np.load(odir+key+'-seps_all.npy') # pix
                                pas_all = np.load(odir+key+'-pas_all.npy') # deg
                                flux_retr_all = np.load(odir+key+'-flux_retr_all.npy') # MJy/sr
                                todo = False
                            except:
                                todo = True
                        else:
                            todo = True
                        
                        # 2D map of the total throughput, i.e., an integration
                        # time weighted average of the coronmsk transmission
                        # over the rolls.
                        tottp = self.get_transmission(pxsc, filt, mask, subarr, odir, key)
                        
                        # The calibrated contrast curves have not been
                        # computed already.
                        if (todo == True):
                            
                            # Offset PSF from WebbPSF, i.e., an integration
                            # time weighted average of the unocculted offset
                            # PSF over the rolls (does account for pupil mask
                            # throughput).
                            offsetpsf = self.get_offsetpsf(filt, mask, key)
                            
                            # Convert the units and compute the injected
                            # fluxes. They need to be in the units of the data
                            # which is MJy/sr.
                            Fstar = self.F0[filt]/10.**(mstar[filt]/2.5)/1e6*np.max(offsetpsf) # MJy; convert the host star brightness from vegamag to MJy
                            if (mask in ['MASKASWB', 'MASKALWB']):
                                cons_inject = np.interp(seps_inject_bar*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                            else:
                                cons_inject = np.interp(seps_inject_rnd*pxsc/1000., seps, cons)*5. # inject at 5 times 5 sigma
                            flux_inject = cons_inject*Fstar*(180./np.pi*3600.*1000.)**2/pxsc**2 # MJy/sr; convert the injected flux from contrast to MJy/sr
                            
                            # If the separation is too small the
                            # interpolation of the contrasts returns nans,
                            # these need to be filtered out before feeding the
                            # fluxes into the injection and recovery routine.
                            good = np.isnan(flux_inject) == False
                            if (mask in ['MASKASWB', 'MASKALWB']):
                                flux_all, seps_all, pas_all, flux_retr_all = self.inject_recover(filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, filt, mask, 10.*fwhm, flux_inject[good], seps_inject_bar[good], pas_inject_bar, KL, ra_off, de_off)
                            else:
                                flux_all, seps_all, pas_all, flux_retr_all = self.inject_recover(filepaths, psflib_filepaths, mode, odir, key, annuli, subsections, pxsc, filt, mask, 10.*fwhm, flux_inject[good], seps_inject_rnd[good], pas_inject_rnd, KL, ra_off, de_off)
                            np.save(odir+key+'-flux_all.npy', flux_all) # MJy/sr
                            np.save(odir+key+'-seps_all.npy', seps_all) # pix
                            np.save(odir+key+'-pas_all.npy', pas_all) # deg
                            np.save(odir+key+'-flux_retr_all.npy', flux_retr_all) # MJy/sr
                        
                        # Group the injection and recovery results by
                        # separation and take the median, then fit a logistic
                        # growth function to them.
                        res = Table([flux_all, seps_all, pas_all, flux_retr_all], names=('flux', 'seps', 'pas', 'flux_retr'))
                        res['tps'] = res['flux_retr']/res['flux']
                        med_res = res.group_by('seps')
                        med_res = med_res.groups.aggregate(np.nanmedian)
                        p0 = np.array([1., 0., 1., 0.2, 15.])
                        pp = least_squares(self.func_lnprob, p0, args=(med_res['seps'], med_res['tps']))
                        # p0 = np.array([0., 1., 1.])
                        # pp = least_squares(self.growth_lnprob, p0, args=(med_res['seps']*pxsc/1000., med_res['tps']))
                        corr_cons = cons/self.func(pp['x'], seps*1000./pxsc)
                        np.save(odir+key+'-pp.npy', pp['x'])
                        
                        # Plot.
                        f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                        extl = (data.shape[1]+1.)/2.*pxsc/1000. # arcsec
                        extr = (data.shape[1]-1.)/2.*pxsc/1000. # arcsec
                        ax[0].imshow(np.log10(np.abs(data[KL])), origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                        for i in range(len(ra_off)):
                            cc = plt.Circle((ra_off[i]/1000., de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[0].add_artist(cc)
                        for i in range(len(flux_all)):
                            ra = seps_all[i]*pxsc*np.sin(np.deg2rad(pas_all[i])) # mas
                            de = seps_all[i]*pxsc*np.cos(np.deg2rad(pas_all[i])) # mas
                            cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
                            ax[0].add_artist(cc)
                        ax[0].set_xlim([5., -5.])
                        ax[0].set_ylim([-5., 5.])
                        ax[0].set_xlabel('$\Delta$RA [arcsec]')
                        ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                        ax[0].set_title('KLIP-subtracted')
                        extl = tottp.shape[1]/2.*pxsc/1000. # arcsec
                        extr = tottp.shape[1]/2.*pxsc/1000. # arcsec
                        p1 = ax[1].imshow(tottp, origin='lower', cmap='viridis', vmin=0., vmax=1., extent=(extl, -extr, -extl, extr))
                        c1 = plt.colorbar(p1, ax=ax[1])
                        c1.set_label('Transmission', rotation=270, labelpad=20)
                        for i in range(len(ra_off)):
                            cc = plt.Circle((ra_off[i]/1000., de_off[i]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[1].add_artist(cc)
                        for i in range(len(flux_all)):
                            ra = seps_all[i]*pxsc*np.sin(np.deg2rad(pas_all[i])) # mas
                            de = seps_all[i]*pxsc*np.cos(np.deg2rad(pas_all[i])) # mas
                            cc = plt.Circle((ra/1000., de/1000.), 10.*pxsc/1000., fill=False, edgecolor='red', linewidth=3)
                            ax[1].add_artist(cc)
                        ax[1].set_xlim([5., -5.])
                        ax[1].set_ylim([-5., 5.])
                        ax[1].set_xlabel('$\Delta$RA [arcsec]')
                        ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                        ax[1].set_title('Transmission')
                        plt.tight_layout()
                        plt.savefig(odir+key+'-cons_inj.pdf')
                        plt.close()
                        
                        # Plot.
                        if (mask in ['MASKASWB', 'MASKALWB']):
                            xx = np.linspace(seps_inject_bar[0], seps_inject_bar[-1], 100)
                        else:
                            xx = np.linspace(seps_inject_rnd[0], seps_inject_rnd[-1], 100)
                        f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                        ax[0].plot(med_res['seps'], med_res['tps'], color='mediumaquamarine', label='Median throughput')
                        ax[0].scatter(res['seps'], res['tps'], s=75, color='mediumaquamarine', alpha=0.5)
                        ax[0].plot(xx, self.func(pp['x'], xx), color='teal', label='Best fit model')
                        ax[0].set_xlim([xx[0], xx[-1]])
                        ax[0].set_ylim([0.0, 1.2])
                        ax[0].grid(axis='y')
                        ax[0].set_xlabel('Separation [pix]')
                        ax[0].set_ylabel('Throughput')
                        ax[0].set_title('Algo & coronmsk throughput')
                        ax[0].legend(loc='lower right')
                        ax[1].plot(seps, cons, color='mediumaquamarine', label='Raw contrast')
                        ax[1].plot(seps, corr_cons, color='teal', label='Calibrated contrast')
                        ax[1].set_yscale('log')
                        ax[1].set_xlim([0., np.max(seps)]) # arcsec
                        ax[1].grid(axis='y')
                        ax[1].set_xlabel('Separation [arcsec]')
                        ax[1].set_ylabel('Contrast [5$\sigma$]')
                        ax[1].set_title('Calibrated contrast curve')
                        ax[1].legend(loc='upper right')
                        plt.tight_layout()
                        plt.savefig(odir+key+'-cons_cal.pdf')
                        plt.close()
                    counter += 1
        
        if (self.verbose == True):
            print('')
        
        return None
    
    def extract_companions(self,
                           mstar, # vegamag
                           ra_off=ra_off, # mas
                           de_off=de_off,
                           KL=-1, # mas
                           overwrite=False):
        """
        Extract astrometry and photometry from any detected companions using
        the pyKLIP forward modeling class.
        
        TODO: tries to convert MJy/sr to contrast using the host star
              magnitude, but the results are wrong by a factor of ~5.
        
        TODO: until RDI is implemented for forward modeling, simply use the
              offset PSF with the correct coronagraphic mask transmission
              applied.
        
        TODO: use a position dependent offset PSF from pyNRC instead of the
              completely unocculted offset PSF from WebbPSF.
        
        Parameters
        ----------
        mstar: dict of float
            Host star magnitude in each filter. Must contain one entry for
            each filter used in the data in the input directory.
        ra_off: list of float
            RA offset of the known companions in the same order as in the
            NIRCCoS config file.
        de_off: list of float
            DEC offset of the known companions in the same order as in the
            NIRCCoS config file.
        KL: int
            Index of the KL component for which the calibrated contrast curve
            and the companion properties shall be computed.
        overwrite: bool
            If true overwrite existing data.
        """
        
        if (self.verbose == True):
            print('--> Extracting companion properties...')
        
        # Loop through all modes, numbers of annuli, and numbers of
        # subsections.
        Nscenarios = len(self.mode)*len(self.annuli)*len(self.subsections)
        counter = 1
        for mode in self.mode:
            for annuli in self.annuli:
                for subsections in self.subsections:
                    
                    # Define the input and output directories for each set of
                    # pyKLIP parameters.
                    idir = self.odir+mode+'_annu%.0f_subs%.0f/FITS/' % (annuli, subsections)
                    odir = self.odir+mode+'_annu%.0f_subs%.0f/FLUX/' % (annuli, subsections)
                    if (not os.path.exists(odir)):
                        os.makedirs(odir)
                    
                    # Create an output directory for the forward modeled datasets.
                    odir_temp = odir+'FITS/'
                    if (not os.path.exists(odir_temp)):
                        os.makedirs(odir_temp)
                    
                    # Loop through all sets of observing parameters.
                    res = {}
                    for i, key in enumerate(self.obs.keys()):
                        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
                        filepaths = np.array(self.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                        ww_cal = np.where(self.obs[key]['TYP'] == 'CAL')[0]
                        psflib_filepaths = np.array(self.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                        hdul = pyfits.open(idir+key+'-KLmodes-all.fits')
                        data = hdul[0].data
                        pxsc = self.obs[key]['PIXSCALE'][0] # mas
                        cent = (hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']) # pix
                        temp = [s.start() for s in re.finditer('_', key)]
                        filt = key[temp[1]+1:temp[2]]
                        mask = key[temp[3]+1:temp[4]]
                        subarr = key[temp[4]+1:]
                        wave = self.wave[filt] # m
                        fwhm = wave/self.diam*rad2mas/pxsc # pix
                        hdul.close()
                        
                        # Create a new pyKLIP dataset for forward modeling the
                        # companion PSFs.
                        dataset = JWST.JWSTData(filepaths=filepaths,
                                                psflib_filepaths=psflib_filepaths)
                        
                        # 2D map of the total throughput, i.e., an integration
                        # time weighted average of the coronmsk transmission
                        # over the rolls.
                        self.get_transmission(pxsc, filt, mask, subarr, odir, key)
                        
                        # Offset PSF from WebbPSF, i.e., an integration time
                        # weighted average of the unocculted offset PSF over
                        # the rolls (does account for pupil mask throughput).
                        offsetpsf = self.get_offsetpsf(filt, mask, key)
                        offsetpsf *= self.F0[filt]/10.**(mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
                        
                        # Loop through all companions.
                        res[key] = {}
                        for j in range(len(ra_off)):
                            
                            # Guesses for the fit parameters.
                            guess_dx = ra_off[j]/pxsc # pix
                            guess_dy = de_off[j]/pxsc # pix
                            guess_sep = np.sqrt(guess_dx**2+guess_dy**2) # pix
                            guess_pa = np.rad2deg(np.arctan2(guess_dx, guess_dy)) # deg
                            guess_flux = 1e-4
                            guess_spec = np.array([1.])
                            
                            # If overwrite is false, check whether the forward
                            # modeled datasets have been computed already.
                            fmdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-fmpsf-KLmodes-all.fits'
                            klipdataset = odir_temp+'FM_C%.0f-' % (j+1)+key+'-klipped-KLmodes-all.fits'
                            if ((overwrite == True) or ((not os.path.exists(fmdataset)) or (not os.path.exists(klipdataset)))):
                                
                                # Initialize the forward modeling pyKLIP class.
                                input_wvs = np.unique(dataset.wvs)
                                if (len(input_wvs) != 1):
                                    raise UserWarning('Only works with broadband photometry')
                                fm_class = fmpsf.FMPlanetPSF(inputs_shape=dataset.input.shape,
                                                             numbasis=np.array(self.truenumbasis[key]),
                                                             sep=guess_sep,
                                                             pa=guess_pa,
                                                             dflux=guess_flux,
                                                             input_psfs=np.array([offsetpsf]),
                                                             input_wvs=input_wvs,
                                                             spectrallib=[guess_spec],
                                                             spectrallib_units='contrast',
                                                             field_dependent_correction=self.correct_transmission)
                                
                                # Compute the forward modeled datasets.
                                annulus = [[guess_sep-20., guess_sep+20.]] # pix
                                subsection = 1
                                fm.klip_dataset(dataset=dataset,
                                                fm_class=fm_class,
                                                mode=mode,
                                                outputdir=odir_temp,
                                                fileprefix='FM_C%.0f-' % (j+1)+key,
                                                annuli=annulus,
                                                subsections=subsection,
                                                movement=1,
                                                numbasis=self.truenumbasis[key],
                                                maxnumbasis=self.maxnumbasis[key],
                                                calibrate_flux=False,
                                                psf_library=dataset.psflib,
                                                highpass=False,
                                                mute_progression=True)
                            
                            # Open the forward modeled datasets.
                            with pyfits.open(fmdataset) as hdul:
                                fm_frame = hdul[0].data[KL]
                                fm_centx = hdul[0].header['PSFCENTX']
                                fm_centy = hdul[0].header['PSFCENTY']
                            with pyfits.open(klipdataset) as hdul:
                                data_frame = hdul[0].data[KL]
                                data_centx = hdul[0].header["PSFCENTX"]
                                data_centy = hdul[0].header["PSFCENTY"]
                            
                            # TODO: until RDI is implemented for forward
                            # modeling, simply use the offset PSF with the
                            # correct coronagraphic mask transmission applied.
                            if (mode == 'RDI'):
                                fm_frame = np.zeros_like(fm_frame)
                                if ((fm_centx % 1. != 0.) or (fm_centy % 1. != 0.)):
                                    raise UserWarning('Requires forward modeled PSF with integer center')
                                else:
                                    fm_centx = int(fm_centx)
                                    fm_centy = int(fm_centy)
                                sx, sy = offsetpsf.shape
                                if ((sx % 2 != 1) or (sy % 2 != 1)):
                                    raise UserWarning('Requires offset PSF with odd shape')
                                temp = shift(offsetpsf.copy(), (guess_dy-int(guess_dy), -(guess_dx-int(guess_dx))))
                                rx = np.arange(sx)-sx//2+guess_dx
                                ry = np.arange(sy)-sy//2+guess_dy
                                xx, yy = np.meshgrid(rx, ry)
                                temp = self.correct_transmission(temp, xx, yy)
                                fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = temp # scale forward modeled PSF similar as in pyKLIP
                                
                                # # FIXME!
                                # # Uncomment the following to use the PSF
                                # # that was injected by pyNRC.
                                # test = []
                                # totet = 0. # s
                                # for ii in range(len(ww_sci)):
                                #     inttm = self.obs[key]['NINTS'][ww_sci[ii]]*self.obs[key]['EFFINTTM'][ww_sci[ii]] # s
                                #     test += [inttm*rotate(np.load('../HIP65426/pynrc_figs/seq_000_filt_'+filt+'_psfs_%+04.0fdeg.npy' % self.obs[key]['PA_V3'][ww_sci[ii]])[j], -self.obs[key]['PA_V3'][ww_sci[ii]], reshape=False, mode='constant', cval=0.)]
                                #     totet += inttm # s
                                # test = np.sum(np.array(test), axis=0)/totet
                                # test *= self.F0[filt]/10.**(mstar[filt]/2.5)/1e6/pxsc**2*(180./np.pi*3600.*1000.)**2 # MJy/sr
                                # fm_frame[fm_centy+int(guess_dy)-sy//2:fm_centy+int(guess_dy)+sy//2+1, fm_centx-int(guess_dx)-sx//2:fm_centx-int(guess_dx)+sx//2+1] = test # scale forward modeled PSF similar as in pyKLIP
                                
                            elif ('RDI' in mode):
                                raise UserWarning('Not implemented yet')
                            
                            # Plot.
                            extl = (fm_frame.shape[1]+1.)/2.*pxsc/1000. # arcsec
                            extr = (fm_frame.shape[1]-1.)/2.*pxsc/1000. # arcsec
                            f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                            p0 = ax[0].imshow(fm_frame*guess_flux, origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                            c0 = plt.colorbar(p0, ax=ax[0])
                            c0.set_label('DN', rotation=270, labelpad=20)
                            cc = plt.Circle((ra_off[j]/1000., de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[0].add_artist(cc)
                            ax[0].set_xlim([5., -5.])
                            ax[0].set_ylim([-5., 5.])
                            ax[0].set_xlabel('$\Delta$RA [arcsec]')
                            ax[0].set_ylabel('$\Delta$DEC [arcsec]')
                            ax[0].set_title(r'FM PSF ($\alpha$ = %.0e)' % guess_flux)
                            p1 = ax[1].imshow(data_frame, origin='lower', cmap='inferno', extent=(extl, -extr, -extl, extr))
                            c1 = plt.colorbar(p1, ax=ax[1])
                            c1.set_label('DN', rotation=270, labelpad=20)
                            cc = plt.Circle((ra_off[j]/1000., de_off[j]/1000.), 10.*pxsc/1000., fill=False, edgecolor='green', linewidth=3)
                            ax[1].add_artist(cc)
                            ax[1].set_xlim([5., -5.])
                            ax[1].set_ylim([-5., 5.])
                            ax[1].set_xlabel('$\Delta$RA [arcsec]')
                            ax[1].set_ylabel('$\Delta$DEC [arcsec]')
                            ax[1].set_title('KLIP-subtracted')
                            plt.tight_layout()
                            plt.savefig(odir+key+'-fmpsf_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Fit the forward modeled PSF to the
                            # KLIP-subtracted data.
                            fitboxsize = 17 # pix
                            fma = fitpsf.FMAstrometry(guess_sep=guess_sep,
                                                      guess_pa=guess_pa,
                                                      fitboxsize=fitboxsize)
                            fma.generate_fm_stamp(fm_image=fm_frame,
                                                  fm_center=[fm_centx, fm_centy],
                                                  padding=5)
                            fma.generate_data_stamp(data=data_frame,
                                                    data_center=[data_centx, data_centy],
                                                    dr=4,
                                                    exclusion_radius=12.*fwhm)
                            corr_len_guess = 3. # pix
                            corr_len_label = r'$l$'
                            fma.set_kernel('matern32', [corr_len_guess], [corr_len_label])                        
                            x_range = 1. # pix
                            y_range = 1. # pix
                            flux_range = 1. # mag
                            corr_len_range = 1. # mag
                            fma.set_bounds(x_range, y_range, flux_range, [corr_len_range])
                            
                            # Make sure that noise map is invertible.
                            noise_map_max = np.nanmax(fma.noise_map)
                            fma.noise_map[np.isnan(fma.noise_map)] = noise_map_max
                            fma.noise_map[fma.noise_map == 0.] = noise_map_max
                            
                            fma.fit_astrometry(nwalkers=100, nburn=200, nsteps=800, numthreads=2)
                            
                            # Plot.
                            chain = fma.sampler.chain
                            f, ax = plt.subplots(4, 1, figsize=(1*6.4, 2*4.8))
                            ax[0].plot(chain[:, :, 0].T, color='black', alpha=1./3.)
                            ax[0].set_xlabel('Steps')
                            ax[0].set_ylabel(r'$\Delta$RA [pix]')
                            ax[1].plot(chain[:, :, 1].T, color='black', alpha=1./3.)
                            ax[1].set_xlabel('Steps')
                            ax[1].set_ylabel(r'$\Delta$DEC [pix]')
                            ax[2].plot(chain[:, :, 2].T, color='black', alpha=1./3.)
                            ax[2].set_xlabel('Steps')
                            ax[2].set_ylabel(r'$\alpha$')
                            ax[3].plot(chain[:, :, 3].T, color='black', alpha=1./3.)
                            ax[3].set_xlabel('Steps')
                            ax[3].set_ylabel(r'$l$ [pix]')
                            plt.suptitle('MCMC chains')
                            plt.tight_layout()
                            plt.savefig(odir+key+'-chains_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Plot.
                            fma.make_corner_plot()
                            plt.savefig(odir+key+'-corner_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Plot.
                            fma.best_fit_and_residuals()
                            plt.savefig(odir+key+'-model_c%.0f' % (j+1)+'.pdf')
                            plt.close()
                            
                            # Write best fit values into results dictionary.
                            temp = 'c%.0f' % (j+1)
                            res[key][temp] = {}
                            res[key][temp]['ra'] = fma.raw_RA_offset.bestfit*pxsc # mas
                            res[key][temp]['dra'] = fma.raw_RA_offset.error*pxsc # mas
                            res[key][temp]['de'] = fma.raw_Dec_offset.bestfit*pxsc # mas
                            res[key][temp]['dde'] = fma.raw_Dec_offset.error*pxsc # mas
                            res[key][temp]['f'] = fma.raw_flux.bestfit
                            res[key][temp]['df'] = fma.raw_flux.error
                            
                            if (self.verbose == True):
                                print('Companion %.0f' % (j+1))
                                print('   RA  = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['ra'], res[key][temp]['dra'], ra_off[j]))
                                print('   DEC = %.2f+/-%.2f mas (%.2f mas guess)' % (res[key][temp]['de'], res[key][temp]['dde'], de_off[j]))
                                try:
                                    condir = self.idir[:self.idir.find('/')]+'/pynrc_figs/'
                                    confiles = [f for f in os.listdir(condir) if filt in f and f.endswith('_cons.npy')]
                                    if (len(confiles) != 1):
                                        raise UserWarning()
                                    else:
                                        con = np.load(condir+confiles[0])[j]
                                    print('   CON = %.2e+/-%.2e (%.2e inj.)' % (res[key][temp]['f'], res[key][temp]['df'], con))
                                except:
                                    print('   CON = %.2e+/-%.2e' % (res[key][temp]['f'], res[key][temp]['df']))
        
        return res
    
    def mask_companions(self,
                        data,
                        pxsc, # mas
                        cent, # pix
                        mrad, # pix
                        ra_off=[], # mas
                        de_off=[]): # mas
        """
        Mask out known companions.
        
        Parameters
        ----------
        data: array
            Data cube of shape (Nframes, Npix, Npix) in which the companions
            shall be masked out.
        pxsc: float
            Pixel scale of the data.
        cent: tuple of float
            Center of the data (i.e., center of the host star PSF).
        mrad: float
            Radius of the mask that is used to mask out the companions.
        ra_off: list of float
            RA offset of the companions that shall be masked out.
        de_off: list of float
            DEC offset of the companions that shall be masked out.
        
        Returns
        -------
        data_masked: array
            Data cube of shape (Nframes, Npix, Npix) in which the companions
            are masked out.
        """
        
        # Sanitize inputs.
        if (len(data.shape) != 3):
            raise UserWarning('Data has invalid shape')
        
        # Mask out known companions.
        data_masked = data.copy()
        yy, xx = np.indices(data.shape[1:]) # pix
        for ra, de in zip(ra_off, de_off):
            dist = np.sqrt((xx-cent[0]+ra/pxsc)**2+(yy-cent[1]-de/pxsc)**2) # pix
            data_masked[:, dist <= mrad] = np.nan
        
        return data_masked
    
    def mask_bar(self,
                 data,
                 cent, # pix
                 pa_ranges_bar=[]): # deg
        """
        Mask out bar mask occulter. This is done by specifying pizza slices
        that shall be considered when computing the contrast curves.
        
        Note: make sure to mask out the bar mask in either of the rolls!
        
        Parameters
        ----------
        data: array
            Data cube of shape (Nframes, Npix, Npix) in which the bar mask
            occulter shall be masked out.
        cent: tuple of float
            Center of the data (i.e., center of the host star PSF).
        pa_ranges_bar: list of tuple of float
            List of tuples defining the pizza slices that shall be considered
            when computing the contrast curves for the bar masks.
        
        Returns
        -------
        data_masked: array
            Data cube of shape (Nframes, Npix, Npix) in which everything but
            the specified pizza slices is masked out.
        """
        
        # Sanitize inputs.
        if (len(data.shape) != 3):
            raise UserWarning('Data has invalid shape')
        
        # Mask out bar mask occulter.
        data_masked = data.copy()
        yy, xx = np.indices(data.shape[1:]) # pix
        tt = np.rad2deg(-1.*np.arctan2((xx-cent[0]), (yy-cent[1]))) % 360. # deg
        for i in range(len(pa_ranges_bar)):
            if (i == 0):
                data_masked[:] = np.nan
            mask = (tt >= pa_ranges_bar[i][0]) & (tt <= pa_ranges_bar[i][1])
            data_masked[:, mask] = data[:, mask]
        
        return data_masked
    
    def correct_transmission(self,
                             stamp,
                             stamp_dx, # pix
                             stamp_dy): # pix
        """
        Apply coronagraphic mask transmission. This uses 2D offset from the
        host star PSF center.
        
        Note: uses a standard cartesian coordinate system so that North is +y
              and East is -x.
        
        Note: uses the 2D transmission map stored in self.transmission. Need
              to run self.get_transmission first!
        
        Parameters
        ----------
        stamp: array
            Frame to which coronagraphic mask transmission shall be applied.
        stamp_dx: array
            Frame of the same shape as stamp containing the x-axis separation
            from the host star PSF center for each pixel.
        stamp_dy: array
            Frame of the same shape as stamp containing the y-axis separation
            from the host star PSF center for each pixel.
        
        Returns
        -------
        stamp: array
            Frame to which coronagraphic mask transmission was applied.
        """
        
        # Apply coronagraphic mask transmission.
        xy = np.vstack((stamp_dy.flatten(), stamp_dx.flatten())).T
        transmission = self.transmission(xy)
        transmission = transmission.reshape(stamp.shape)
        
        return transmission*stamp
    
    def get_transmission(self,
                         pxsc, # mas
                         filt,
                         mask,
                         subarr,
                         odir,
                         key):
        """
        Write coronagraphic mask transmission into self.transmission. The
        output is a 2D transmission map containing the derotated and
        integration time weighted average of the PSF masks from CRDS.
        
        TODO: assumes that (159.5, 159.5) is the center of the PSF masks from
              CRDS. This seems to be true for the round masks. For the bar
              masks, this needs to be confirmed. Then, uses the PSF position
              with respect to the NRCA4_MASKSWB and the NRCA5_MASKLWB subarray
              from pySIAF to shift the bar mask PSF masks to their new center.
        
        Parameters
        ----------
        pxsc: float
            Pixel scale of the PSF masks from CRDS.
        filt: str
            Filter name from JWST data header.
        mask: str
            Coronagraphic mask name from JWST data header.
        subarr: str
            Subarray name from JWST data header.
        odir: str
            Output directory for the plots.
        key: str
            Dictionary key of the self.obs dictionary specifying the
            considered observation.
        
        Returns
        -------
        tottp: array
            2D transmission map containing the derotated and integration time
            weighted average of the PSF masks from CRDS.
        """
        
        # Check if the fiducial point override is active.
        if ((fiducial_point_override == True) and (mask in ['MASKASWB', 'MASKALWB'])):
            filt_temp = 'narrow'
        else:
            filt_temp = filt
        
        # Find the science target observations.
        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
        
        # Open the correct PSF mask. Download it from CRDS if it is not yet
        # in the psfmaskdir.
        psfmask = self.psfmask[filt_temp+'_'+mask+'_'+subarr]
        if (not os.path.exists(psfmaskdir)):
            os.makedirs(psfmaskdir)
        if (not os.path.exists(psfmaskdir+psfmask+'.fits')):
            urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmask+'.fits', psfmaskdir+psfmask+'.fits')
        hdul = pyfits.open(psfmaskdir+psfmask+'.fits')
        tp = hdul['SCI'].data[1:-1, 1:-1] # crop artifact at the edge
        hdul.close()
        
        # Shift the bar mask PSF masks to their new center. Values outside of
        # the subarray are filled with zeros (i.e., no transmission).
        if (mask in ['MASKASWB']):
            tp = shift(tp, (0., -self.offset_swb[filt_temp]*1000./pxsc), mode='constant', cval=0.)
        elif (mask in ['MASKALWB']):
            tp = shift(tp, (0., -self.offset_lwb[filt_temp]*1000./pxsc), mode='constant', cval=0.)
        
        # Compute the derotated and integration time weighted average of the
        # PSF masks. Values outside of the subarray are filled with zeros
        # (i.e., no transmission). Then, create a regular grid interpolator
        # taking 2D pixel offset as input and returning the coronagraphic mask
        # transmission.
        ramp = np.arange(tp.shape[0]) # pix
        xx, yy = np.meshgrid(ramp, ramp) # pix
        xx = xx-158.5 # pix; new center because PSF mask was cropped by 2 pixel
        yy = yy-158.5 # pix; new center because PSF mask was cropped by 2 pixel
        dist = np.sqrt(xx**2+yy**2) # pix
        tottp = np.zeros_like(tp)
        totet = 0. # s
        for i in range(len(ww_sci)):
            inttm = self.obs[key]['NINTS'][ww_sci[i]]*self.obs[key]['EFFINTTM'][ww_sci[i]] # s
            tottp += inttm*rotate(tp.copy(), -self.obs[key]['PA_V3'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
            totet += inttm # s
        tottp /= totet
        tottp[dist > self.owa] = np.nan
        self.transmission = RegularGridInterpolator((xx[0, :], yy[:, 0]), tottp)
        
        # Plot.
        plt.figure(figsize=(6.4, 4.8))
        ax = plt.gca()
        pp = ax.imshow(tottp, origin='lower', cmap='viridis', vmin=0, vmax=1)
        cc = plt.colorbar(pp, ax=ax)
        cc.set_label('Transmission', rotation=270, labelpad=20)
        ax.set_xlabel('$\Delta$RA [pix]')
        ax.set_ylabel('$\Delta$DEC [pix]')
        ax.set_title('Transmission')
        plt.tight_layout()
        plt.savefig(odir+key+'-transmission.pdf')
        plt.close()
        
        return tottp
    
    def correct_transmission_old(self,
                                 stamp,
                                 stamp_dx, # pix
                                 stamp_dy): # pix
        """
        """
        
        dist = np.sqrt(stamp_dx**2+stamp_dy**2) # pix
        transmission = np.interp(dist, self.transmission[0], self.transmission[1])
        transmission = transmission.reshape(stamp.shape)
        
        return transmission*stamp
    
    def get_transmission_old(self,
                             pxsc, # mas
                             filt,
                             mask,
                             subarr,
                             odir,
                             key):
        """
        """
        
        if ((fiducial_point_override == True) and (mask in ['MASKASWB', 'MASKALWB'])):
            filt_temp = 'narrow'
        else:
            filt_temp = filt
        
        if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
            with open(transmissiondir+mask+'.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                sep1 = [] # pix
                trm1 = []
                for row in reader:
                    sep1 += [float(row[0])/pxsc*1000.] # pix
                    trm1 += [float(row[1])]
                sep1 = np.array(sep1) # pix
                trm1 = np.array(trm1)
            ww = sep1 > self.owa
            sep1 = np.delete(sep1, ww) # pix
            trm1 = np.delete(trm1, ww)
        elif (mask in ['MASKASWB']):
            with open(transmissiondir+'MASKSWB_F182M.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                sep1 = [] # pix
                trm1 = []
                for row in reader:
                    sep1 += [float(row[0])/pxsc*1000.] # pix
                    trm1 += [float(row[1])]
                sep1 = np.array(sep1) # pix
                trm1 = np.array(trm1)
            xa = self.offset_swb['F182M'] # arcsec
            xb = self.offset_swb['F212N'] # arcsec
            xx = self.offset_swb[filt_temp] # arcsec
            wa = self.wave['F182M'] # m
            wb = self.wave['F212N'] # m
            func = interp1d(np.array([xa, xb]), np.array([wa, wb]), fill_value='extrapolate')
            wx = func(xx) # m
            sep1 *= wx/wa # pix
            ww = sep1 > self.owa
            sep1 = np.delete(sep1, ww) # pix
            trm1 = np.delete(trm1, ww)
        elif (mask in ['MASKALWB']):
            with open(transmissiondir+'MASKLWB_F250M.csv', newline='') as csvfile:
                reader = csv.reader(csvfile)
                sep1 = [] # pix
                trm1 = []
                for row in reader:
                    sep1 += [float(row[0])/pxsc*1000.] # pix
                    trm1 += [float(row[1])]
                sep1 = np.array(sep1) # pix
                trm1 = np.array(trm1)
            xa = self.offset_lwb['F250M'] # arcsec
            xb = self.offset_lwb['F480M'] # arcsec
            xx = self.offset_lwb[filt_temp] # arcsec
            wa = self.wave['F250M'] # m
            wb = self.wave['F480M'] # m
            func = interp1d(np.array([xa, xb]), np.array([wa, wb]), fill_value='extrapolate')
            wx = func(xx) # m
            sep1 *= wx/wa # pix
            ww = sep1 > self.owa
            sep1 = np.delete(sep1, ww) # pix
            trm1 = np.delete(trm1, ww)
        else:
            raise UserWarning()
        
        if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
            psfmask = self.psfmask[filt_temp+'_'+mask+'_'+subarr]
            if (not os.path.exists(psfmaskdir)):
                    os.makedirs(psfmaskdir)
            if (not os.path.exists(psfmaskdir+psfmask+'.fits')):
                urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmask+'.fits', psfmaskdir+psfmask+'.fits')
            hdul = pyfits.open(psfmaskdir+psfmask+'.fits')
            sep2, trm2 = ot.azimuthalAverage(hdul['SCI'].data.copy(), center=(159.5, 159.5), returnradii=True, binsize=1)
            ww = sep2 > self.owa
            sep2 = np.delete(sep2, ww) # pix
            trm2 = np.delete(trm2, ww)
            hdul.close()
        elif (mask in ['MASKASWB']):
            psfmask = self.psfmask[filt_temp+'_'+mask+'_'+subarr]
            if (not os.path.exists(psfmaskdir)):
                    os.makedirs(psfmaskdir)
            if (not os.path.exists(psfmaskdir+psfmask+'.fits')):
                urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmask+'.fits', psfmaskdir+psfmask+'.fits')
            hdul = pyfits.open(psfmaskdir+psfmask+'.fits')
            sz = hdul['SCI'].data.shape[0]
            xx = self.offset_swb[filt_temp]*1000./pxsc+(sz-1.)/2. # pix
            xa = int(np.floor(xx)) # pix
            xb = int(np.ceil(xx)) # pix
            trma = hdul['SCI'].data[sz//2:, xa]
            trmb = hdul['SCI'].data[sz//2:, xb]
            trm2 = trma+(trmb-trma)*(xx-xa)
            sep2 = np.linspace(0.5, (sz-1.)/2., sz//2) # pix
            ww = sep2 > self.owa
            sep2 = np.delete(sep2, ww) # pix
            trm2 = np.delete(trm2, ww)
            hdul.close()
        elif (mask in ['MASKALWB']):
            psfmask = self.psfmask[filt_temp+'_'+mask+'_'+subarr]
            if (not os.path.exists(psfmaskdir)):
                    os.makedirs(psfmaskdir)
            if (not os.path.exists(psfmaskdir+psfmask+'.fits')):
                urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/'+psfmask+'.fits', psfmaskdir+psfmask+'.fits')
            hdul = pyfits.open(psfmaskdir+psfmask+'.fits')
            sz = hdul['SCI'].data.shape[0]
            xx = self.offset_lwb[filt_temp]*1000./pxsc+(sz-1.)/2. # pix
            xa = int(np.floor(xx)) # pix
            xb = int(np.ceil(xx)) # pix
            trma = hdul['SCI'].data[sz//2:, xa]
            trmb = hdul['SCI'].data[sz//2:, xb]
            trm2 = trma+(trmb-trma)*(xx-xa)
            sep2 = np.linspace(0.5, (sz-1.)/2., sz//2) # pix
            ww = sep2 > self.owa
            sep2 = np.delete(sep2, ww) # pix
            trm2 = np.delete(trm2, ww)
            hdul.close()
        else:
            raise UserWarning()
        
        plt.figure(figsize=(6.4, 4.8))
        ax = plt.gca()
        ax.plot(sep1, trm1, label='JDox')
        ax.plot(sep2, trm2, label='PSF mask')
        ax.set_xlim([0., self.owa]) # pix
        ax.grid(axis='y')
        ax.set_xlabel('Separation [pix]')
        ax.set_ylabel('Transmission')
        ax.set_title('Transmission')
        ax.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig(odir+key+'-transmission.pdf')
        plt.close()
        
        if (use_psfmask == True):
            self.transmission = (sep2, trm2)
        else:
            self.transmission = (sep1, trm1)
        
        return None
    
    def get_offsetpsf(self,
                      filt,
                      mask,
                      key):
        """
        Get a derotated and integration time weighted average of an offset PSF
        from WebbPSF. Try to load it from the offsetpsfdir and generate it if
        it is not in there yet. This offset PSF is normalized to an integrated
        source of 1 and takes into account the throughput of the pupil mask.
        
        Parameters
        ----------
        filt: str
            Filter name from JWST data header.
        mask: str
            Coronagraphic mask name from JWST data header.
        key: str
            Dictionary key of the self.obs dictionary specifying the
            considered observation.
        
        Returns
        -------
        totop: array
            Stamp of the derotated and integration time weighted average of
            the offset PSF.
        """
        
        # Try to load the offset PSF from the offsetpsfdir and generate it if
        # it is not in there yet.
        if (not os.path.exists(offsetpsfdir)):
            os.makedirs(offsetpsfdir)
        if (not os.path.exists(offsetpsfdir+filt+'_'+mask+'.npy')):
            self.gen_offsetpsf(filt, mask)
        offsetpsf = np.load(offsetpsfdir+filt+'_'+mask+'.npy')
        
        # Find the science target observations.
        ww_sci = np.where(self.obs[key]['TYP'] == 'SCI')[0]
        
        # Compute the derotated and integration time weighted average of the
        # offset PSF. Values outside of the PSF stamp are filled with zeros.
        totop = np.zeros_like(offsetpsf)
        totet = 0. # s
        for i in range(len(ww_sci)):
            inttm = self.obs[key]['NINTS'][ww_sci[i]]*self.obs[key]['EFFINTTM'][ww_sci[i]] # s
            totop += inttm*rotate(offsetpsf.copy(), -self.obs[key]['PA_V3'][ww_sci[i]], reshape=False, mode='constant', cval=0.)
            totet += inttm # s
        totop /= totet
        
        return totop
    
    def gen_offsetpsf(self,
                      filt,
                      mask):
        """
        Generate an offset PSF using WebbPSF. This offset PSF is normalized to
        an integrated source of 1 and takes into account the throughput of the
        pupil mask.
        
        Parameters
        ----------
        filt: str
            Filter name from JWST data header.
        mask: str
            Coronagraphic mask name from JWST data header.
        """
        
        # Apply the correct pupil mask, but no image mask (unocculted PSF).
        nircam.filter = filt
        if (mask in ['MASKA210R', 'MASKA335R', 'MASKA430R']):
            nircam.pupil_mask = 'MASKRND'
        elif (mask in ['MASKASWB']):
            nircam.pupil_mask = 'MASKSWB'
        elif (mask in ['MASKALWB']):
            nircam.pupil_mask = 'MASKLWB'
        else:
            raise UserWarning()
        nircam.image_mask = None
        
        # Compute the offset PSF using WebbPSF and save it to the offsetpsfdir.
        hdul = nircam.calc_psf(oversample=1)
        psf = hdul[0].data # PSF center is at (39, 39)
        hdul.close()
        np.save(offsetpsfdir+filt+'_'+mask+'.npy', psf)
        
        return None
    
    def inject_recover(self,
                       filepaths,
                       psflib_filepaths,
                       mode,
                       odir,
                       key,
                       annuli,
                       subsections,
                       pxsc, # mas
                       filt,
                       mask,
                       mrad, # pix
                       flux_inject=[], # MJy/sr
                       seps_inject=[], # pix
                       pas_inject=[], # deg
                       KL=-1,
                       ra_off=[], # mas
                       de_off=[]): # mas
        """
        Inject fake companions and recover them by fitting a 2D Gaussian.
        Makes sure that the injected fake companions are not too close to any
        real companions and also that fake companions which are too close to
        each other are injected into different fake datasets.
        
        TODO: recover fake companions by fitting an offset PSF instead of a 2D
              Gaussian.
        
        TODO: currently uses WebbPSF to compute a theoretical offset PSF. It
              should be possible to use PSF stamps extracted from the
              astrometric confirmation images in the future.
        
        TODO: use a position dependent offset PSF from pyNRC instead of the
              completely unocculted offset PSF from WebbPSF.
        
        Parameters
        ----------
        filepaths: list of str
            List of stage 2 reduced science target files for the considered
            observation.
        psflib_filepaths: list of str
            List of stage 2 reduced reference target files for the considered
            observation.
        mode: list of str
            List of modes for pyKLIP, will loop through all.
        odir: str
            Output directory for the plots and pyKLIP products.
        key: str
            Dictionary key of the self.obs dictionary specifying the
            considered observation.
        annuli: list of int
            List of number of annuli for pyKLIP, will loop through all.
        subsections: list of int
            List of number of subsections for pyKLIP, will loop through all.
        pxsc: float
            Pixel scale of the data.
        filt: str
            Filter name from JWST data header.
        mask: str
            Coronagraphic mask name from JWST data header.
        mrad: float
            Radius of the mask that is used to mask out the companions.
        flux_inject: list of float
            List of fluxes at which fake planets shall be injected.
        seps_inject: list of float
            List of separations at which fake planets shall be injected.
        pas_inject: list of float
            List of position angles at which fake planets shall be injected.
        KL: int
            Index of the KL component for which the fake companions shall be
            injected.
        ra_off: list of float
            RA offset of the companions that shall be masked out.
        de_off: list of float
            DEC offset of the companions that shall be masked out.
        
        Returns
        -------
        
        """
        
        # Sanitize inputs.
        Nfl = len(flux_inject)
        Nse = len(seps_inject)
        Npa = len(pas_inject)
        if (Nfl != Nse):
            raise UserWarning('Injected fluxes need to match injected separations')
        
        # Create an output directory for the datasets with fake companions.
        odir_temp = odir+'FITS/'
        if (not os.path.exists(odir_temp)):
            os.makedirs(odir_temp)
        
        # Offset PSF from WebbPSF, i.e., an integration time weighted average
        # of the unocculted offset PSF over the rolls (normalized to a peak
        # flux of 1).
        offsetpsf = self.get_offsetpsf(filt, mask, key)
        offsetpsf /= np.max(offsetpsf)
        
        # Initialize outputs.
        flux_all = [] # MJy/sr
        seps_all = [] # pix
        pas_all = [] # deg
        flux_retr_all = [] # MJy/sr
        done = []
        
        # Do not inject fake companions closer than mrad to any known
        # companion.
        for i in range(Nse):
            for j in range(Npa):
                ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
                de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
                for k in range(len(ra_off)):
                    dist = np.sqrt((ra-ra_off[k])**2+(de-de_off[k])**2) # mas
                    if (dist < mrad*pxsc):
                        done += [i*Npa+j]
                        break
        
        # If not finished yet, create a new pyKLIP dataset into which fake
        # companions will be injected.
        finished = False
        ctr = 0
        while (finished == False):
            dataset = JWST.JWSTData(filepaths=filepaths,
                                    psflib_filepaths=psflib_filepaths)
            
            # Inject fake companions. Make sure that no other fake companion
            # closer than mrad will be injected into the same dataset.
            todo = []
            for i in range(Nse):
                for j in range(Npa):
                    if (i*Npa+j not in done):
                        ra = seps_inject[i]*pxsc*np.sin(np.deg2rad(pas_inject[j])) # mas
                        de = seps_inject[i]*pxsc*np.cos(np.deg2rad(pas_inject[j])) # mas
                        flag = True
                        for k in range(len(todo)):
                            jj = todo[k] % Npa
                            ii = (todo[k]-jj)//Npa
                            ra_temp = seps_inject[ii]*pxsc*np.sin(np.deg2rad(pas_inject[jj])) # mas
                            de_temp = seps_inject[ii]*pxsc*np.cos(np.deg2rad(pas_inject[jj])) # mas
                            dist = np.sqrt((ra-ra_temp)**2+(de-de_temp)**2) # mas
                            if (dist < mrad*pxsc):
                                flag = False
                                break
                        if (flag == True):
                            todo += [i*Npa+j]
                            done += [i*Npa+j]
                            stamp = np.array([offsetpsf*flux_inject[i] for k in range(dataset.input.shape[0])]) # MJy/sr
                            fakes.inject_planet(frames=dataset.input, centers=dataset.centers, inputflux=stamp, astr_hdrs=dataset.wcs, radius=seps_inject[i], pa=pas_inject[j], field_dependent_correction=self.correct_transmission)
                            flux_all += [flux_inject[i]] # MJy/sr
                            seps_all += [seps_inject[i]] # pix
                            pas_all += [pas_inject[j]] # deg
            
            # Run pyKLIP.
            parallelized.klip_dataset(dataset=dataset,
                                      mode=mode,
                                      outputdir=odir_temp,
                                      fileprefix='FAKE_%04.0f_' % ctr+key,
                                      annuli=annuli,
                                      subsections=subsections,
                                      movement=1,
                                      numbasis=[self.truenumbasis[key][KL]],
                                      calibrate_flux=False,
                                      maxnumbasis=self.maxnumbasis[key],
                                      psf_library=dataset.psflib,
                                      highpass=False,
                                      verbose=False)
            
            # Recover fake companions by fitting a 2D Gaussian.
            klipdataset = odir_temp+'FAKE_%04.0f_' % ctr+key+'-KLmodes-all.fits'
            with pyfits.open(klipdataset) as hdul:
                outputfile = hdul[0].data[0]
                outputfile_centers = [hdul[0].header['PSFCENTX'], hdul[0].header['PSFCENTY']]
            for k in range(len(todo)):
                jj = todo[k] % Npa
                ii = (todo[k]-jj)//Npa
                fake_flux = fakes.retrieve_planet_flux(frames=outputfile, centers=outputfile_centers, astr_hdrs=dataset.output_wcs[0], sep=seps_inject[ii], pa=pas_inject[jj], searchrad=5)
                flux_retr_all += [fake_flux] # MJy/sr
            ctr += 1
            
            # Check if finished, i.e., if all fake companions were injected.
            # If not, continue with a new dataset.
            if (len(done) == Nse*Npa):
                finished = True
        
        # Make outputs arrays.
        flux_all = np.array(flux_all) # MJy/sr
        seps_all = np.array(seps_all) # pix
        pas_all = np.array(pas_all) # deg
        flux_retr_all = np.array(flux_retr_all) # MJy/sr
        
        return flux_all, seps_all, pas_all, flux_retr_all
    
    def func(self,
             p,
             x):
        
        y = p[0]*(1.-np.exp(-(x-p[1])**2/(2*p[2]**2)))*(1-p[3]*np.exp(-(x-p[4])**2/(2*p[2]**2)))
        y[x < p[1]] = 0.
        
        return y
    
    def func_lnprob(self,
                    p,
                    x,
                    y):
        
        return np.abs(y[:-1]-self.func(p, x[:-1]))


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':

    proc = processor(idir,
                     odir,
                     mode=mode,
                     annuli=annuli,
                     subsections=subsections,
                     numbasis=numbasis,
                     verbose=verbose)
    proc.run_pyklip()

    proc.raw_contrast_curve(mstar,
                            ra_off=ra_off,
                            de_off=de_off,
                            pa_ranges_bar=pa_ranges_bar)

    proc.cal_contrast_curve(mstar,
                            ra_off=ra_off,
                            de_off=de_off,
                            seps_inject_rnd=seps_inject_rnd,
                            pas_inject_rnd=pas_inject_rnd,
                            seps_inject_bar=seps_inject_bar,
                            pas_inject_bar=pas_inject_bar,
                            KL=KL,
                            overwrite=False)

    res = proc.extract_companions(mstar,
                                  ra_off=ra_off,
                                  de_off=de_off,
                                  overwrite=True)

    print('DONE')
