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

import pyklip.klip

from astropy import wcs
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline
from pyklip import parallelized, rdi
from pyklip.instruments.Instrument import Data

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class SpaceTelescope(Data):
    """
    The pyKLIP instrument class for space telescope data.
    """
    
    ####################
    ### Constructors ###
    ####################
    
    def __init__(self,
                 obs,
                 filepaths,
                 psflib_filepaths=None):
        
        # Initialize pyKLIP Data class.
        super(SpaceTelescope, self).__init__()
        
        # Read science and reference files.
        self.readdata(obs, filepaths)
        if psflib_filepaths is not None:
            self.readpsflib(obs, psflib_filepaths)
        else:
            self._psflib = None
        
        pass
    
    ################################
    ### Instance Required Fields ###
    ################################
    
    @property
    def input(self):
        return self._input
    @input.setter
    def input(self, newval):
        self._input = newval
    
    @property
    def centers(self):
        return self._centers
    @centers.setter
    def centers(self, newval):
        self._centers = newval
    
    @property
    def filenums(self):
        return self._filenums
    @filenums.setter
    def filenums(self, newval):
        self._filenums = newval
    
    @property
    def filenames(self):
        return self._filenames
    @filenames.setter
    def filenames(self, newval):
        self._filenames = newval
    
    @property
    def PAs(self):
        return self._PAs
    @PAs.setter
    def PAs(self, newval):
        self._PAs = newval
    
    @property
    def wvs(self):
        return self._wvs
    @wvs.setter
    def wvs(self, newval):
        self._wvs = newval
    
    @property
    def wcs(self):
        return self._wcs
    @wcs.setter
    def wcs(self, newval):
        self._wcs = newval
    
    @property
    def IWA(self):
        return self._IWA
    @IWA.setter
    def IWA(self, newval):
        self._IWA = newval
    
    @property
    def psflib(self):
        return self._psflib
    @psflib.setter
    def psflib(self, newval):
        self._psflib = newval
    
    @property
    def output(self):
        return self._output
    @output.setter
    def output(self, newval):
        self._output = newval
    
    ###############
    ### Methods ###
    ###############
    
    def readdata(self,
                 obs,
                 filepaths):
        
        # Check input.
        if isinstance(filepaths, str):
            filepaths = np.array([filepaths])
        if len(filepaths) == 0:
            raise UserWarning('No science files provided to pyKLIP')
        
        # Loop through science files.
        input_all = []
        centers_all = [] # pix
        filenames_all = []
        PAs_all = [] # deg
        wvs_all = [] # m
        wcs_all = []
        PIXSCALE = [] # mas
        for i, filepath in enumerate(filepaths):
            
            # Read science file.
            hdul = pyfits.open(filepath)
            TELESCOP = hdul[0].header['TELESCOP']
            data = hdul['SCI'].data
            pxdq = hdul['DQ'].data
            if data.ndim == 2:
                data = data[np.newaxis, :]
                pxdq = pxdq[np.newaxis, :]
            if data.ndim != 3:
                raise UserWarning('Requires 2D/3D data cube')
            NINTS = data.shape[0]
            ww = np.where(obs['FITSFILE'] == filepath)[0][0]
            
            # Nan out non-science pixels.
            data[pxdq & 512 == 512] = np.nan
            
            # Get image centers.
            centers = np.array([obs['CRPIX1'][ww] - 1 + obs['XOFFSET'][ww] / obs['PIXSCALE'][ww], obs['CRPIX2'][ww] - 1 + obs['YOFFSET'][ww] / obs['PIXSCALE'][ww]] * NINTS)
            
            # Get metadata.
            input_all += [data]
            centers_all += [centers]
            filenames_all += [os.path.split(filepath)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            PAs_all += [obs['ROLL_REF'][ww]] * NINTS
            wvs_all += [1e-6 * obs['CWAVEL'][ww]] * NINTS
            wcs_hdr = wcs.WCS(header=hdul['SCI'].header, naxis=hdul['SCI'].header['WCSAXES'])
            for j in range(NINTS):
                wcs_all += [wcs_hdr.deepcopy()]
            PIXSCALE += [obs['PIXSCALE'][ww]]
            hdul.close()
        input_all = np.concatenate(input_all)
        if input_all.ndim != 3:
            raise UserWarning('Some science files do not have matching image shapes')
        centers_all = np.concatenate(centers_all).reshape(-1, 2)
        filenames_all = np.array(filenames_all)
        filenums_all = np.array(range(len(filenames_all)))
        PAs_all = np.array(PAs_all)
        wvs_all = np.array(wvs_all)
        wcs_all = np.array(wcs_all)
        PIXSCALE = np.unique(np.array(PIXSCALE))
        if len(PIXSCALE) != 1:
            raise UserWarning('Some science files do not have matching pixel scales')
        if TELESCOP == 'JWST' and obs['EXP_TYPE'][ww] in ['NRC_CORON', 'MIR_4QPM', 'MIR_LYOT']:
            iwa_all = np.min(wvs_all) / 6.5 * 180. / np.pi * 3600. * 1000. / PIXSCALE[0] # pix
        else:
            iwa_all = 1. # pix
        
        # Recenter science images.
        new_center = np.array(data.shape[1:]) / 2.
        new_center = new_center[::-1]
        for i, image in enumerate(input_all):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=centers_all[i])
            input_all[i] = recentered_image
            centers_all[i] = new_center
        
        # Assign pyKLIP variables.
        self._input = input_all
        self._centers = centers_all
        self._filenames = filenames_all
        self._filenums = filenums_all
        self._PAs = PAs_all
        self._wvs = wvs_all
        self._wcs = wcs_all
        self._IWA = iwa_all
        
        pass
    
    def readpsflib(self,
                   obs,
                   psflib_filepaths):
        
        # Check input.
        if isinstance(psflib_filepaths, str):
            psflib_filepaths = np.array([psflib_filepaths])
        if len(psflib_filepaths) == 0:
            raise UserWarning('No reference files provided to pyKLIP')
        
        # Loop through reference files.
        psflib_data_all = []
        psflib_centers_all = [] # pix
        psflib_filenames_all = []
        for i, filepath in enumerate(psflib_filepaths):
            
            # Read reference file.
            hdul = pyfits.open(filepath)
            data = hdul['SCI'].data
            pxdq = hdul['DQ'].data
            if data.ndim == 2:
                data = data[np.newaxis, :]
                pxdq = pxdq[np.newaxis, :]
            if data.ndim != 3:
                raise UserWarning('Requires 2D/3D data cube')
            NINTS = data.shape[0]
            ww = np.where(obs['FITSFILE'] == filepath)[0][0]
            
            # Nan out non-science pixels.
            data[pxdq & 512 == 512] = np.nan
            
            # Get image centers.
            centers = np.array([obs['CRPIX1'][ww] - 1 + obs['XOFFSET'][ww] / obs['PIXSCALE'][ww], obs['CRPIX2'][ww] - 1 + obs['YOFFSET'][ww] / obs['PIXSCALE'][ww]] * NINTS)
            
            # Get metadata.
            psflib_data_all += [data]
            psflib_centers_all += [centers]
            psflib_filenames_all += [os.path.split(filepath)[1] + '_INT%.0f' % (j + 1) for j in range(NINTS)]
            hdul.close()
        psflib_data_all = np.concatenate(psflib_data_all)
        if psflib_data_all.ndim != 3:
            raise UserWarning('Some reference files do not have matching image shapes')
        psflib_centers_all = np.concatenate(psflib_centers_all).reshape(-1, 2)
        psflib_filenames_all = np.array(psflib_filenames_all)
        
        # Recenter reference images.
        new_center = np.array(data.shape[1:]) / 2.
        new_center = new_center[::-1]
        for i, image in enumerate(psflib_data_all):
            recentered_image = pyklip.klip.align_and_scale(image, new_center=new_center, old_center=psflib_centers_all[i])
            psflib_data_all[i] = recentered_image
            psflib_centers_all[i] = new_center
        
        # Append science data.
        psflib_data_all = np.append(psflib_data_all, self._input, axis=0)
        psflib_centers_all = np.append(psflib_centers_all, self._centers, axis=0)
        psflib_filenames_all = np.append(psflib_filenames_all, self._filenames, axis=0)
        
        # Initialize PSF library.
        psflib = rdi.PSFLibrary(psflib_data_all, new_center, psflib_filenames_all, compute_correlation=True)
        
        # Prepare PSF library.
        psflib.prepare_library(self)
        
        # Assign pyKLIP variables.
        self._psflib = psflib
        
        pass
    
    def savedata(self,
                 filepath,
                 data,
                 klipparams=None,
                 filetype='',
                 zaxis=None,
                 more_keywords=None):
        
        # Make FITS file.
        hdul = pyfits.HDUList()
        hdul.append(pyfits.PrimaryHDU(data))
        
        # Write all used files to header. Ignore duplicates.
        filenames = np.unique(self.filenames)
        Nfiles = np.size(filenames)
        hdul[0].header['DRPNFILE'] = (Nfiles, 'Num raw files used in pyKLIP')
        error = False
        for i, filename in enumerate(filenames):
            try:
                hdul[0].header['FILE_{0}'.format(i)] = filename + '.fits'
            except:
                if not error:
                    log.warning('--> Too many files to be written to header, skipping')
                    error = True
        
        # Write PSF subtraction parameters and pyKLIP version to header.
        try:
            pyklipver = pyklip.__version__
        except:
            pyklipver = 'unknown'
        hdul[0].header['PSFSUB'] = ('pyKLIP', 'PSF Subtraction Algo')
        hdul[0].header.add_history('Reduced with pyKLIP using commit {0}'.format(pyklipver))
        hdul[0].header['CREATOR'] = 'pyKLIP-{0}'.format(pyklipver)
        hdul[0].header['pyklipv'] = (pyklipver, 'pyKLIP version that was used')
        if klipparams is not None:
            hdul[0].header['PSFPARAM'] = (klipparams, 'KLIP parameters')
            hdul[0].header.add_history('pyKLIP reduction with parameters {0}'.format(klipparams))
        
        # Write z-axis units to header if necessary.
        if zaxis is not None:
            if 'KL Mode' in filetype:
                hdul[0].header['CTYPE3'] = 'KLMODES'
                for i, klmode in enumerate(zaxis):
                    hdul[0].header['KLMODE{0}'.format(i)] = (klmode, 'KL Mode of slice {0}'.format(i))
                hdul[0].header['CUNIT3'] = 'N/A'
                hdul[0].header['CRVAL3'] = 1
                hdul[0].header['CRPIX3'] = 1.
                hdul[0].header['CD3_3'] = 1.
        
        # Write WCS information to header.
        wcshdr = self.output_wcs[0].to_header()
        for key in wcshdr.keys():
            hdul[0].header[key] = wcshdr[key]
        
        # Write extra keywords to header if necessary.
        if more_keywords is not None:
            for hdr_key in more_keywords:
                hdul[0].header[hdr_key] = more_keywords[hdr_key]
        
        # Update image center.
        center = self.output_centers[0]
        hdul[0].header.update({'PSFCENTX': center[0], 'PSFCENTY': center[1]})
        hdul[0].header.update({'CRPIX1': center[0], 'CRPIX2': center[1]})
        hdul[0].header.add_history('Image recentered to {0}'.format(str(center)))
        
        # Write FITS file.
        try:
            hdul.writeto(filepath, overwrite=True)
        except TypeError:
            hdul.writeto(filepath, clobber=True)
        hdul.close()
        
        pass

def run_obs(Database,
            kwargs={},
            subdir='klipsub'):
    
    # Check input.
    if 'mode' not in kwargs.keys():
        kwargs['mode'] = ['ADI+RDI']
    if not isinstance(kwargs['mode'], list):
        kwargs['mode'] = [kwargs['mode']]
    if 'annuli' not in kwargs.keys():
        kwargs['annuli'] = [1]
    if not isinstance(kwargs['annuli'], list):
        kwargs['annuli'] = [kwargs['annuli']]
    if 'subsections' not in kwargs.keys():
        kwargs['subsections'] = [1]
    if not isinstance(kwargs['subsections'], list):
        kwargs['subsections'] = [kwargs['subsections']]
    if 'numbasis' not in kwargs.keys():
        kwargs['numbasis'] = [1, 2, 5, 10, 20, 50, 100]
    if not isinstance(kwargs['numbasis'], list):
        kwargs['numbasis'] = [kwargs['numbasis']]
    kwargs_temp = kwargs.copy()
    if 'movement' not in kwargs_temp.keys():
        kwargs_temp['movement'] = 1.
    kwargs_temp['calibrate_flux'] = False
    if 'verbose' not in kwargs_temp.keys():
        kwargs_temp['verbose'] = Database.verbose
    
    # Set output directory.
    output_dir = os.path.join(Database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    kwargs_temp['outputdir'] = output_dir
    
    # Loop through concatenations.
    datapaths = []
    for i, key in enumerate(Database.obs.keys()):
        log.info('--> Concatenation ' + key)
        
        # Find science and reference files.
        filepaths = []
        psflib_filepaths = []
        first_sci = True
        nints = []
        Nfitsfiles = len(Database.obs[key])
        for j in range(Nfitsfiles):
            if Database.obs[key]['TYPE'][j] == 'SCI':
                filepaths += [Database.obs[key]['FITSFILE'][j]]
                if first_sci:
                    first_sci = False
                else:
                    nints += [Database.obs[key]['NINTS'][j]]
            elif Database.obs[key]['TYPE'][j] == 'REF':
                psflib_filepaths += [Database.obs[key]['FITSFILE'][j]]
                nints += [Database.obs[key]['NINTS'][j]]
        filepaths = np.array(filepaths)
        psflib_filepaths = np.array(psflib_filepaths)
        nints = np.array(nints)
        maxnumbasis = np.sum(nints)
        if 'maxnumbasis' not in kwargs_temp.keys() or kwargs_temp['maxnumbasis'] is None:
            kwargs_temp['maxnumbasis'] = maxnumbasis
        
        # Initialize pyKLIP dataset.
        dataset = SpaceTelescope(Database.obs[key], filepaths, psflib_filepaths)
        kwargs_temp['dataset'] = dataset
        kwargs_temp['aligned_center'] = dataset.psflib.aligned_center
        kwargs_temp['psf_library'] = dataset.psflib
        
        # Run KLIP subtraction.
        for mode in kwargs['mode']:
            for annu in kwargs['annuli']:
                for subs in kwargs['subsections']:
                    log.info('  --> pyKLIP: mode = ' + mode + ', annuli = ' + str(annu) + ', subsections = ' + str(subs))
                    fileprefix = mode + '_NANNU' + str(annu) + '_NSUBS' + str(subs) + '_' + key
                    kwargs_temp['fileprefix'] = fileprefix
                    kwargs_temp['mode'] = mode
                    kwargs_temp['annuli'] = annu
                    kwargs_temp['subsections'] = subs
                    parallelized.klip_dataset(**kwargs_temp)
                    
                    # Get reduction path.
                    datapath = os.path.join(output_dir, fileprefix + '-KLmodes-all.fits')
                    datapaths += [datapath]
                    
                    # Update reduction header.
                    ww_sci = np.where(Database.obs[key]['TYPE'] == 'SCI')[0]
                    hdul = pyfits.open(datapath)
                    hdul[0].header['TELESCOP'] = Database.obs[key]['TELESCOP'][ww_sci[0]]
                    hdul[0].header['TARGPROP'] = Database.obs[key]['TARGPROP'][ww_sci[0]]
                    hdul[0].header['TARG_RA'] = Database.obs[key]['TARG_RA'][ww_sci[0]]
                    hdul[0].header['TARG_DEC'] = Database.obs[key]['TARG_DEC'][ww_sci[0]]
                    hdul[0].header['INSTRUME'] = Database.obs[key]['INSTRUME'][ww_sci[0]]
                    hdul[0].header['DETECTOR'] = Database.obs[key]['DETECTOR'][ww_sci[0]]
                    hdul[0].header['FILTER'] = Database.obs[key]['FILTER'][ww_sci[0]]
                    hdul[0].header['CWAVEL'] = Database.obs[key]['CWAVEL'][ww_sci[0]]
                    hdul[0].header['DWAVEL'] = Database.obs[key]['DWAVEL'][ww_sci[0]]
                    hdul[0].header['PUPIL'] = Database.obs[key]['PUPIL'][ww_sci[0]]
                    hdul[0].header['CORONMSK'] = Database.obs[key]['CORONMSK'][ww_sci[0]]
                    hdul[0].header['EXP_TYPE'] = Database.obs[key]['EXP_TYPE'][ww_sci[0]]
                    hdul[0].header['EXPSTART'] = np.min(Database.obs[key]['EXPSTART'][ww_sci])
                    hdul[0].header['NINTS'] = np.sum(Database.obs[key]['NINTS'][ww_sci])
                    hdul[0].header['EFFINTTM'] = Database.obs[key]['EFFINTTM'][ww_sci[0]]
                    hdul[0].header['SUBARRAY'] = Database.obs[key]['SUBARRAY'][ww_sci[0]]
                    hdul[0].header['APERNAME'] = Database.obs[key]['APERNAME'][ww_sci[0]]
                    hdul[0].header['PIXSCALE'] = Database.obs[key]['PIXSCALE'][ww_sci[0]]
                    hdul[0].header['MODE'] = mode
                    hdul[0].header['ANNULI'] = annu
                    hdul[0].header['SUBSECTS'] = subs
                    hdul[0].header['BUNIT'] = Database.obs[key]['BUNIT'][ww_sci[0]]
                    hdul.writeto(datapath, output_verify='fix', overwrite=True)
                    hdul.close()
    
    # Read reductions into database.
    Database.read_jwst_s3_data(datapaths)
    
    pass
