import os, glob
from jwst.pipeline.calwebb_image2 import Image2Pipeline
from jwst.outlier_detection.outlier_detection_step import OutlierDetectionStep


from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, generic_filter

from pyklip.instruments.JWST import trim_miri_data

from . import utils
from . import io

def run_image_processing(meta, subdir_str, itype):
    files = io.get_working_files(meta, meta.done_rampfit, subdir=subdir_str, search=meta.imgproc_ext, itype=itype)

    # Create save directory
    save_dir = meta.odir + subdir_str.replace('RAMPFIT', 'IMGPROCESS')
    if os.path.exists(save_dir) == False:
        os.makedirs(save_dir)

    # Run the pipeline on every file
    if meta.outlier_only != True:
        for file in files:
            # Set up pipeline
            pipeline = Image2Pipeline()
            pipeline.output_dir = save_dir
            pipeline.logcfg = pipeline.output_dir + 'imageprocess-log.cfg'
            pipeline.save_results = True
            pipeline.run(file)

    # Perform additional cleaning
    if meta.outlier_corr != 'None':

        # Make a folder to save to
        clean_savedir = save_dir + '_CLEAN'
        if os.path.exists(clean_savedir) == False:
            os.makedirs(clean_savedir)

        # Get the files from the image processing step only if it is completed
        if meta.outlier_only != True:
            files = glob.glob(save_dir+'/*')

        #Use the JWST outlier detections step to flag a few more bad pixels.
        if hasattr(meta,'jwst_outlier_detection'):
            if meta.jwst_outlier_detection:
                step = OutlierDetectionStep()
                for file in files:
                    outDataModel = step.process(file)
                    outDataModel.save(clean_savedir+"/"+outDataModel.meta.filename)
                files = glob.glob(clean_savedir+'/*')

        for file in files:
            with fits.open(file) as hdu:
                # Grab the data
                filt = hdu[0].header['FILTER']
                inst = hdu[0].header['INSTRUME']

                raw_data = hdu['SCI'].data
                raw_dq = hdu['DQ'].data

                # Trim data if needed
                if inst == 'MIRI':
                    data_trim, trim = trim_miri_data([raw_data, raw_dq], filt)
                    data = data_trim[0] # Only one cube so just want first index
                    dq = data_trim[1]
                else:
                    data = raw_data
                    dq = raw_dq

                    # print(file)
                    # from matplotlib.colors import LogNorm
                    # plt.imshow(dq[0], norm=LogNorm())
                    # plt.show()

                # Clean each image of outlier bad pixels
                if 'bpclean' in meta.outlier_corr:
                    # Default to sigclip=5 and niter=5 if attributes not found
                    sigclip = meta.clean_sigclip if hasattr(meta, 'clean_sigclip') else 5
                    niter   = meta.outlier_niter if hasattr(meta, 'outlier_niter') else 5
                    data = utils.clean_data(data, dq, sigclip=sigclip, niter=niter)

                if 'median' in meta.outlier_corr:
                    for i, arr in enumerate(data):
                        blurred = median_filter(arr, size=meta.outmed_blur) # Blur image
                        diff = np.subtract(arr, blurred)    #Take difference
                        threshold = meta.outmed_threshold*np.std(diff)  # Set outlier threshold
                        outliers = np.nonzero((np.abs(diff[1:-1,1:-1])>threshold)) # Find outliers
                        outliers = np.array(outliers) + 1 #Because we trimmed in line above


                        cleaned = np.copy(arr)
                        for y,x in zip(outliers[0], outliers[1]):
                            cleaned[y,x] = blurred[y,x] #Swap pixels with blurred image

                        # arry, arrx = cleaned.shape
                        #Make the center normal
                        # cleaned[int(3*arry/5):int(4*arry/5),int(3*arrx/5):int(4*arrx/5)] = \
                        #   arr[int(3*arry/5):int(4*arry/5),int(3*arrx/5):int(4*arrx/5)]

                        data[i] = cleaned
                if 'timemed' in meta.outlier_corr:
                    z, y, x = data.shape
                    # datacopy = np.copy(data)
                    for row in range(y):
                        for col in range(x):
                            if inst == 'MIRI':
                                pix_time = data[1:,row,col] #Don't use first image for this
                            else:
                                pix_time = data[0:,row,col]

                            blurred = median_filter(pix_time, size=5)
                            diff = np.subtract(pix_time, blurred)
                            threshold = 2*np.std(diff)
                            outliers = np.nonzero(np.abs(diff)>threshold)

                            cleaned = np.copy(pix_time)
                            # mask = np.copy(pix_time)
                            for j in outliers:
                                cleaned[j] = blurred[j] #Swap pixels with blurred image
                            if inst == 'MIRI':
                                data[1:,row,col] = cleaned
                            else:
                                data[0:,row,col] = cleaned
                if 'timemed_alt' in meta.outlier_corr:
                    z, y, x = data.shape
                    # datacopy = np.copy(data)
                    for row in range(y):
                        for col in range(x):
                            pix_time = data[0:,row,col] #Don't use first image for this
                            fracs = pix_time / np.max(pix_time)

                            # Look for things that were bright, but change
                            if np.max(pix_time) > 1 and np.min(fracs) < meta.tmed_alt_frac:
                                good = np.where(fracs < meta.tmed_alt_frac)
                                bad = np.where(fracs > meta.tmed_alt_frac)
                                pix_time[bad] = np.median(pix_time[good])
                                cleaned = pix_time
                            else:
                                cleaned = pix_time

                            data[0:,row,col] = cleaned

                if 'dqmed' in meta.outlier_corr:
                    for i, arr in enumerate(data):
                        baddq = np.argwhere(dq[i]>meta.dq_threshold)
                        cleaned = np.copy(arr)
                        for pix in baddq:
                            if pix[0] > 2 and pix[1] > 2 and pix[0]<arr.shape[0]-2 and pix[1]<arr.shape[1]-2:
                                ylo, yhi = pix[0]-1, pix[0]+2
                                xlo, xhi = pix[1]-1, pix[1]+2
                                sub = arr[ylo:yhi, xlo:xhi]
                                if len(sub != 0):
                                    sub[1,1] = np.nan
                                    cleaned[pix[0],pix[1]] = np.nanmedian(sub)
                        data[i] = cleaned

                if 'custom' in meta.outlier_corr:
                    if not hasattr(meta, 'custom_file'):
                        print('meta.custom_file attribute not set. Skipping custom outlier fixing.')
                    elif meta.custom_file is None:
                        print('meta.custom_file set to None. Skipping custom outlier fixing.')
                    else:
                        badpix = np.loadtxt(meta.custom_file, delimiter=',', dtype=int)
                        if inst == 'MIRI':
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

                #Assign to original array
                if inst == 'MIRI':
                    shape = data.shape
                    hdu['SCI'].data[:,trim[1]:trim[1]+shape[1],trim[0]:trim[0]+shape[2]] = data
                else:
                    hdu['SCI'].data[:] = data
                hdu.writeto(clean_savedir+'/'+file.split('/')[-1], overwrite=True)

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


from jwst.associations.load_as_asn import LoadAsLevel2Asn
from jwst import datamodels

class Coron2Pipeline(Image2Pipeline):
    """
    Coron2Pipeline: Processes JWST imaging-mode slope data from Level-2a to
    Level-2b.

    Included steps are:
    background_subtraction, assign_wcs, flat_field, photom, and pixel cleaning.
    The resample step is skipped for coronagraphic exposure types.
    """

    class_alias = "calwebb_coron2"

    spec = """
        outlier_type = string(default='bpclean') # Outlier cleaning algorithm(s)
        outmed_blur = integr(default=5) # Size of median blurring window for 'median' 
        outmed_threshold = integer(default=1) # Threshold to detect outliers for 'median'
        dq_threshold = integer(default=1000) # Min DQ Mask value for 'dqmed'
        tmed_alt_frac = float(default=0.2) # Fractional value for 'timemed_alt'
        custom_file = string(default='') # Custom file for 'custom'
    """

    def __init__(self, *args, **kwargs):

        # Add jwst outlier detection step 
        self.step_defs['outlier_detection'] = OutlierDetectionStep

        Image2Pipeline.__init__(self, *args, **kwargs)

    def process(self, input):

        self.log.info(f'Starting {self.class_alias} ...')

        # Retrieve the input(s)
        asn = LoadAsLevel2Asn.load(input, basename=self.output_file)

        # Each exposure is a product in the association.
        # Process each exposure.
        results = []
        for product in asn['products']:
            self.log.info('Processing product {}'.format(product['name']))
            if self.save_results:
                self.output_file = product['name']
            try:
                getattr(asn, 'filename')
            except AttributeError:
                asn.filename = "singleton"

            result = self.process_exposure_product(
                product,
                asn['asn_pool'],
                os.path.basename(asn.filename)
            )

            # Pixel cleaning
            result = self.outlier_detection(result)
            result = self.clean_images(result)

            # Save result
            suffix = 'cal'
            if isinstance(result, datamodels.CubeModel):
                suffix = 'calints'
            result.meta.filename = self.make_output_path(suffix=suffix)
            results.append(result)

        self.log.info(f'... ending {self.class_alias}')

        self.output_use_model = True
        self.suffix = False
        return results

    def trim_data(self, model):
        """Returns trimmed data, dq mask, and trim indices for MIRI (no change for others)"""

        # Trim data if needed
        inst = model.meta.instrument.name
        if inst == 'MIRI':
            filt = model.meta.instrument.filter
            data_trim, trim = trim_miri_data([model.data, model.dq], filt)
            # data, dq, trim
            return data_trim[0], data_trim[1], trim
        else:
            data = model.data
            dq = model.dq
            return model.data, model.dq, None

    def clean_images(self, model, **kwargs):
        """Perform bad pixel fixing on flagged and outlier pixels."""

        inst = model.meta.instrument.name
        data, dq, trim = self.trim_data(self, model)

        if 'bpclean' in self.outlier_type:
            data = self.clean_bpfix(data, dq, **kwargs)
        if 'median' in self.outlier_type:
            data = self.clean_median(data, **kwargs)
        if 'timemed_orig' in self.outlier_type:
            data = self.clean_timemed(data)
        if 'timemed_alt' in self.outlier_type:
            data = self.clean_timemed_alt(data, **kwargs)
        if 'dqmed' in self.outlier_type:
            data = self.clean_dqmed(data, dq, **kwargs)
        if 'custom' in self.outlier_type:
            data = self.clean_custom(data, trim=trim, **kwargs)

        #Assign to original array
        if inst == 'MIRI':
            shape = data.shape
            model.data[:,trim[1]:trim[1]+shape[1],trim[0]:trim[0]+shape[2]] = data
        else:
            model.data = data

        return model

    def clean_bpfix(self, data, dq, **kwargs):
        "Clean using bpclean / bpfix algorithm"

        self.log.info('Performing bpfix cleaning algorithm')
        data = utils.clean_data(data, dq, in_place=True, **kwargs)
        self.log.info('... Completed bpfix cleaning.')
        return data

    def clean_median(self, data, blur=None, threshold=None, **kwargs):
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

            # Create a series of shifted images
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
            self.log.warn('custom_file attribute is not set. Skipping custom outlier fixing.')
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


