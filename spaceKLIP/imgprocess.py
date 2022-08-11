import os, glob

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.ndimage import median_filter, generic_filter

from pyklip.instruments.JWST import trim_miri_data

from jwst import datamodels
from jwst.stpipe import Step
from jwst.pipeline.calwebb_image2 import Image2Pipeline
from jwst.associations.load_as_asn import LoadAsLevel2Asn
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep

# Define logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

from . import utils
from . import io

class CleanPixStep(Step):
    """Clean outlier bad pixels of input data"""

    class_alias = "outlier_clean"

    spec = """
        outlier_type = string(default='bpclean') # Outlier cleaning algorithms
        bpclean_sigclip = integer(default=5)     # Pixel sigma clipping for 'bpclean'
        bpclean_niter = integer(default=5)       # Number of iterations to perform 'bpclean'
        outmed_blur = integer(default=5)         # Size of median blurring window for 'median' 
        outmed_threshold = integer(default=1)    # Threshold to detect outliers for 'median'
        dq_threshold = integer(default=1000)     # Min DQ Mask value for 'dqmed'
        tmed_alt_frac = float(default=0.2)       # Fractional value for 'timemed_alt'
        custom_file = string(default='')         # Custom file for 'custom'
    """

    def process(self, input_data):

        with datamodels.open(input_data, save_open=False) as input_model:
            self.input_model = input_model

            # Check if CubeModel or ImageModel type
            if not isinstance(self.input_model, (datamodels.CubeModel, datamodels.ImageModel)):
                raise TypeError('Input must be compatible with CubeModel or ImageModel datamodel.')

            self.clean_images()

            return self.input_model

            
    def clean_images(self):
        """Perform bad pixel fixing on flagged and outlier pixels."""

        # Reform data and dq array into cubes if input is ImageModel
        if isinstance(self.input_model, datamodels.ImageModel):
            data = self.input_model.data[np.newaxis,:,:]
            dq = self.input_model.dq[np.newaxis,:,:]
        else:
            data = self.input_model.data
            dq = self.input_model.dq

        # Trim data if MIRI
        inst = self.input_model.meta.instrument.name
        if inst == 'MIRI':
            filt = self.input_model.meta.instrument.filter
            data_trim, trim = trim_miri_data([data, dq], filt)
            data, dq = data_trim
        else:
            trim = [0,0]

        # These functions expect input data to be cubes
        if 'bpclean' in self.outlier_type:
            kwargs = {
                'sigclip' : self.bpclean_sigclip,
                'niter'   : self.bpclean_niter,
            }
            data = self.clean_bpfix(data, dq, **kwargs)

        if 'median' in self.outlier_type:
            kwargs = {
                'blur'      : self.outmed_blur,
                'threshold' : self.outmed_threshold,
            }
            data = self.clean_median(data, **kwargs)

        if ('timemed' in self.outlier_type) and ('timemed_alt' not in self.outlier_type):
            data = self.clean_timemed(data, inst)

        if 'timemed_alt' in self.outlier_type:
            data = self.clean_timemed_alt(data, tmed_alt_frac=self.tmed_alt_frac)

        if 'dqmed' in self.outlier_type:
            data = self.clean_dqmed(data, dq, dq_threshold=self.dq_threshold)

        if 'custom' in self.outlier_type:
            data = self.clean_custom(data, trim=trim, file=self.custom_file)

        # Assign back to original array
        nz, ny, nx = data.shape
        x1 = trim[0]
        x2 = x1 + nx
        y1 = trim[1]
        y2 = y1 + ny

        # Is input model 2D or 3D?
        if isinstance(self.input_model, datamodels.ImageModel):
            self.input_model.data[y1:y2, x1:x2] = data[0]
        else:
            self.input_model.data[:, y1:y2, x1:x2] = data

    def clean_bpfix(self, data, dq, sigclip=None, niter=None, **kwargs):
        """Clean using bpclean / bpfix algorithm
        
        Parameters
        ==========
        data : ndarray
            Image cube.
        dq : ndarray
            Data quaity array same size as data.
        
        Keyword Args
        ============
        sigclip : int
            How many sigma from mean doe we fix?
        niter : int
            How many iterations for sigma clipping? 
            Ignored if bpmask is set.
        in_place : bool
            Do in-place corrections of input array.
            Otherwise, works on a copy.
        pix_shift : int
            Size of border pixels to compare to.
            We find bad pixels by comparing to neighbors and replacing.
            E.g., if set to 1, use immediate adjacents neighbors.
            Replaces with a median of surrounding pixels.
        rows : bool
            Compare to row pixels? Setting to False will ignore pixels
            along rows during comparison. Recommended to increase
            ``pix_shift`` parameter if using only rows or cols.
        cols : bool
            Compare to column pixels? Setting to False will ignore pixels
            along columns during comparison. Recommended to increase
            ``pix_shift`` parameter if using only rows or cols.
        verbose : bool
            Print number of fixed pixels per iteration
        """

        self.log.info('Performing bpclean/bpfix cleaning algorithm')

        sigclip = self.bpclean_sigclip if sigclip is None else sigclip
        niter   = self.bpclean_niter   if niter   is None else niter

        data = utils.clean_data(data, dq, in_place=True, sigclip=sigclip, niter=niter, **kwargs)

        self.log.info('... Completed bpclean/bpfix cleaning.')

        return data

    def clean_median(self, data, blur=None, threshold=None):
        """Blur image with median filter"""

        self.log.info('Performing median cleaning algorithm')

        blur      = self.outmed_blur      if blur      is None else blur
        threshold = self.outmed_threshold if threshold is None else threshold

        # Blur all frames in cube along (y,x) axes
        blurred_cube = median_filter(data, size=(1,blur,blur))

        #Take difference
        diff = data - blurred_cube  
        # Set outlier threshold for each image
        # TODO: Use outlier-resistante function (e.g., medabsdev)?
        nsig_thresh = threshold * np.std(diff, axis=(-2,-1))
        # Create outlier mask
        outlier_mask = np.zeros_like(diff).astype('bool')
        outlier_mask[:,1:-1,1:-1] = np.abs(diff[:,1:-1,1:-1]) > nsig_thresh.reshape([-1,1,1])

        # Update outlier data with blurred data
        data[outlier_mask] = blurred_cube[outlier_mask]

        self.log.info('... Completed median cleaning.')
        return data

    def clean_timemed(self, raw_data, inst):
        """Median clean of pixels along time dimension
        
        Doesn't work very well with only two images.
        """

        self.log.info('Performing timemed cleaning algorithm')

        # Exclude first integration for MIRI
        data = raw_data[1:] if inst=='MIRI' else raw_data
        nz = data.shape[0]

        # Blur images along time axis (hardcoded to 5 pixels or nz)
        size = (5,1,1) if nz>=5 else (nz,1,1)
        blurred_cube = median_filter(data, size=size)
        diff = data - blurred_cube
        # Get nsigma threshold for flagging outliers (hardcoded to 2-sigma)
        # TODO: Outlie resistant sigma?
        nsig_thresh = 2*np.std(diff)
        # Flag pixels above threshold
        outlier_mask = np.abs(diff) > nsig_thresh
        # Replace data with pixels from median image
        data[outlier_mask] = blurred_cube[outlier_mask]

        self.log.info('... Completed timemed cleaning.')
        if inst == 'MIRI':
            raw_data[1:] = data
            return raw_data
        else:
            return data

    def clean_timemed_alt(self, data, tmed_alt_frac=None):
        """Search for variance in bright pixels"""

        self.log.info('Performing timemed_alt cleaning algorithm')

        tmed_alt_frac = self.tmed_alt_frac if tmed_alt_frac is None else tmed_alt_frac

        max_vals = np.max(data, axis=0)
        fracs = data / max_vals

        # Look for bright pixels with large variations
        mask = (max_vals > 1) & (np.min(fracs, axis=0) < tmed_alt_frac)
        data_mask = data[:,mask].copy()
        # Flag those values with large variance and set to NaN
        ibad = fracs[:,mask] > tmed_alt_frac
        data_mask[ibad] = np.nan
        # Get the median of all pixels, excluding the NaNs
        data_med = np.nanmedian(data_mask, axis=0)
        # Replace NaN'ed pixels with it's median value
        for i, im in enumerate(data_mask):
            ind_nan = np.isnan(im)
            data_mask[i,ind_nan] = data_med[ind_nan]

        # Replace model data
        data[:,mask] = data_mask

        self.log.info('... Completed timemed_alt cleaning.')
        return data

    def clean_dqmed(self, data, dq, dq_threshold=None):
        """Replace DQ Mask pixels with median of surroundings"""

        self.log.info('Performing dqmed cleaning algorithm')

        dq_threshold = self.dq_threshold if dq_threshold is None else dq_threshold

        for i, im in enumerate(data):
            dq_im = dq[i]
            # Mask of all flagged pixels
            baddq = dq_im > dq_threshold

            # Exclude border pixels from fixing routine
            baddq[0:2,:] = False
            baddq[-2:,:] = False
            baddq[:,0:2] = False
            baddq[:,-2:] = False

            # Set all flagged pixels to NaN's
            im[baddq] = np.nan

            # Create a series of shifted images to median over
            shift_arr = []
            sh_vals = [-1,0,1]
            for ix in sh_vals:
                for iy in sh_vals:
                    if (ix != 0) or (iy != 0): # skip case of (0,0)
                        shift_arr.append(np.roll(im, (iy,ix), axis=(0,1)))
            shift_arr = np.asarray(shift_arr)
            # Replace bad pixels with median of good pixels
            im[baddq] = np.nanmedian(shift_arr, axis=0)[baddq]

            # Save into data
            data[i] = im

        self.log.info('... Completed dqmed cleaning.')

        return data

    def clean_custom(self, data, trim=None, file=None):
        """Replace custom pixels with median of surroundings"""

        self.log.info('Performing custom pixel cleaning algorithm ...')

        file = self.custom_file if file is None else file
        if file=='':
            self.log.warning('custom_file attribute is not set. Skipping custom outlier fixing.')
        else:
            badpix = np.loadtxt(file, delimiter=',', dtype=int)
            if trim is not None:
                badpix -= trim # To account for trimming of MIRI array

            badpix -= [1,1] #To account for DS9 offset
            for i, arr in enumerate(data):
                cleaned = np.copy(arr)
                for pix in badpix:  
                    if pix[0] > 1 and pix[1] > 1 and pix[0]<arr.shape[0]-2 and pix[1]<arr.shape[1]-2:
                        ylo, yhi = pix[1]-1, pix[1]+2
                        xlo, xhi = pix[0]-1, pix[0]+2
                        sub = arr[ylo:yhi, xlo:xhi]
                        if len(sub != 0):
                            sub[1,1] = np.nan
                            cleaned[pix[1],pix[0]] = np.nanmedian(sub)
                data[i] = cleaned

        self.log.info('... Completed custom pixel cleaning.')

        return data

class Coron2Pipeline(Image2Pipeline):
    """
    Coron2Pipeline: Processes JWST imaging-mode slope data from Level-2a to
    Level-2b.

    Cleaned data is saved as *cal.fit or *calints.fits into a new directory
    with _CLEAN appended by default.

    Included steps are:
    background_subtraction, assign_wcs, flat_field, photom, and pixel cleaning.
    The resample step is skipped for coronagraphic exposure types.
    """

    class_alias = "calwebb_coron2"

    spec = """
        find_outliers = boolean(default=True) # Flag additional pixels with OutlierDetectionStep
        clean_data = boolean(default=True)    # Perform pixel cleaning
        clean_only = boolean(default=False)   # Perform cleaning on existing cal files
    """

    def __init__(self, *args, **kwargs):

        # Add jwst outlier detection step 
        self.step_defs['outlier_step'] = OutlierDetectionStep
        self.step_defs['clean_step'] = CleanPixStep

        Image2Pipeline.__init__(self, *args, **kwargs)

        if self.clean_only and not self.clean_data:
            self.log.info('clean_only is specified, but clean_data was set to False. Setting clean_data=True.')
            self.clean_data = True

    def process(self, input):

        self.log.info(f'Starting {self.class_alias} ...')

        # Retrieve the input(s)
        asn = LoadAsLevel2Asn.load(input, basename=self.output_file)

        # Each exposure is a product in the association.
        # Process each exposure.
        all_results = []
        for product in asn['products']:
            self.log.info('Processing product {}'.format(product['name']))
            if self.save_results:
                self.output_file = product['name']
            try:
                getattr(asn, 'filename')
            except AttributeError:
                asn.filename = "singleton"

            filebase = os.path.basename(asn.filename)
            if self.clean_only:
                # Finding existing cal / calints for cleaning
                result = self.find_exposure_product(product, asn['asn_pool'], filebase)
            else:
                result = self.process_exposure_product(product, asn['asn_pool'], filebase)

            if self.find_outliers:
                result = self.outlier_step.run(result)

            if self.clean_data:
                result = self.clean_step(result)

            # Save result
            suffix = 'cal'
            if isinstance(result, datamodels.CubeModel):
                suffix = 'calints'
            result.meta.filename = self.make_output_path(suffix=suffix)
            all_results.append(result)

        self.log.info(f'... ending {self.class_alias}')

        self.output_use_model = True
        self.suffix = False
        return all_results


    # Process each exposure
    def find_exposure_product(self, exp_product, pool_name=' ', asn_file=' '):
        """Process an exposure found in the association product

        Parameters
        ---------
        exp_product: dict
            A Level2b association product.

        pool_name: str
            The pool file name. Used for recording purposes only.

        asn_file: str
            The name of the association file.
            Used for recording purposes only.
        """
        from collections import defaultdict

        # Find all the member types in the product
        members_by_type = defaultdict(list)
        for member in exp_product['members']:
            fname = member['expname']
            # Replace rate / rateints with cal / calints
            fname = member['expname'].replace('rate', 'cal')
            members_by_type[member['exptype'].lower()].append(fname)

        # Get the science member. Technically there should only be
        # one. We'll just get the first one found.
        science = members_by_type['science']
        if len(science) != 1:
            wstr = 'Wrong number of science files found in {}'.format(exp_product['name'])
            self.log.warning(wstr)
            self.log.warning('    Using only first one.')
        science = science[0]

        self.log.info('Working on input %s ...', science)
        if isinstance(science, datamodels.DataModel):
            input = science
        else:
            input = datamodels.open(science)

        # Record ASN pool and table names in output
        input.meta.asn.pool_name = pool_name
        input.meta.asn.table_name = asn_file

        # That's all folks
        self.log.info(
            'Finished locating product {}'.format(exp_product['name'])
        )
        return input



def run_image_processing(meta, subdir_str, itype):
    files = io.get_working_files(meta, meta.done_rampfit, subdir=subdir_str, search=meta.imgproc_ext, itype=itype)

    # Create save directory
    save_dir = meta.odir + subdir_str.replace('RAMPFIT', 'IMGPROCESS')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # Type of outlier correction / cleaning requested
    if hasattr(meta, 'outlier_corr') and ((meta.outlier_corr is not None) or (meta.outlier_corr.lower() != 'none')):
        outlier_type = meta.outlier_corr
    else:
        outlier_type = ''

    # Performing cleaning on existing cal data?
    do_clean_only = do_clean = False
    if hasattr(meta, 'outlier_only') and meta.outlier_only:
        do_clean_only = True
        do_clean = True  # Must be set to True
    if outlier_type != '':
        do_clean = True

    # Some consistency checks...
    if do_clean and outlier_type == '':
        raise ValueError("Pixel cleaning requested, but outlier_corr has not been set!")

    # Create clean directory
    if do_clean:
        clean_savedir = save_dir + '_CLEAN'
        if os.path.exists(clean_savedir) == False:
            os.makedirs(clean_savedir)
        output_dir = clean_savedir
    else:
        output_dir = save_dir

    # Update input files if do_clean_only
    if do_clean_only:
        subdir_str = subdir_str.replace('RAMPFIT', 'IMGPROCESS')
        try:
            # Get calints to be cleaned
            files_clean = io.get_working_files(meta, meta.done_rampfit, subdir=subdir_str, search=meta.sub_ext, itype=itype)
        except ValueError:
            log.warning('meta.outlier_only was set, but no existing calints found! Generating calints first...')

            # Run basic Stage2 pipeline if no cal files present for outlier_only=True
            for file in files:
                logger, fh = io.open_new_log_file(file, save_dir, stage_str='image2')
                pipeline = Coron2Pipeline(
                    output_dir=save_dir, 
                    clean_data=False, 
                    clean_only=False, 
                    find_outliers = False,
                )

                pipeline.save_results = True
                
                # Run pipeline, raise exception on error, and close log file handler
                try:
                    pipeline.run(file)
                except Exception as e:
                    raise RuntimeError(
                        'Caught exception during pipeline processing.'
                        '\nException: {}'.format(e)
                    )
                finally:
                    pipeline.closeout()
                    io.close_log_file(logger, fh)

            log.info('Completed calints. Continuing...')
            # Get calints to be cleaned
            files_clean = io.get_working_files(meta, meta.done_rampfit, subdir=subdir_str, search=meta.sub_ext, itype=itype)
        finally:
            # Set files at end
            files = files_clean

    #####################################
    # Run pipeline
    for file in files:
        # Create a new log file and file stream handler for logging
        logger, fh = io.open_new_log_file(file, output_dir, stage_str='image2')

        # Set up pipeline
        pipeline = Coron2Pipeline(
            output_dir=output_dir, 
            clean_data=do_clean,  # Perform cleaning?
            clean_only=do_clean_only, # Perform cleaning on already existing cal files?
        )

        # pipeline.logcfg = pipeline.output_dir + 'imageprocess-log.cfg'

        # Perform JWST OulierDetection Step?
        if hasattr(meta,'jwst_outlier_detection'):
            pipeline.find_outliers = meta.jwst_outlier_detection

        # Update clean step attributes
        pipeline.clean_step.outlier_type = outlier_type
        if hasattr(meta, 'bpclean_sigclip'):
            pipeline.clean_step.bpclean_sigclip = meta.bpclean_sigclip
        if hasattr(meta, 'bpclean_niter'):
            pipeline.clean_step.bpclean_niter = meta.bpclean_niter
        if hasattr(meta, 'outmed_blur'):
            pipeline.clean_step.outmed_blur = meta.outmed_blur
        if hasattr(meta, 'outmed_threshold'):
            pipeline.clean_step.outmed_threshold = meta.outmed_threshold
        if hasattr(meta, 'dq_threshold'):
            pipeline.clean_step.dq_threshold = meta.dq_threshold
        if hasattr(meta, 'tmed_alt_frac'):
            pipeline.clean_step.tmed_alt_frac = meta.tmed_alt_frac
        if hasattr(meta, 'custom_file'):
            pipeline.clean_step.custom_file = meta.custom_file

        pipeline.save_results = True

        # Run pipeline, raise exception on error, and close log file handler
        try:
            pipeline.run(file)
        except Exception as e:
            raise RuntimeError(
                'Caught exception during pipeline processing.'
                '\nException: {}'.format(e)
            )
        finally:
            pipeline.closeout()
            io.close_log_file(logger, fh)

    return

def stsci_image_processing(meta):
    """
    Use the JWST pipeline to process *rateints.fits files to *calints.fits files
    """
    if meta.imgproc_idir:
        run_image_processing(meta, 'RAMPFIT/SCI+REF', itype='default')
    if meta.imgproc_bgdirs:
        if meta.bg_sci_dir != 'None':
            run_image_processing(meta, 'RAMPFIT/BGSCI', itype='bgsci')
        if meta.bg_ref_dir != 'None':
            run_image_processing(meta, 'RAMPFIT/BGREF', itype='bgref')

    return

