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
                if ('timemed' in meta.outlier_corr) and ('timemed_alt' not in meta.outlier_corr):
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



