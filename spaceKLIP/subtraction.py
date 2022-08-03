import glob, os, sys

import numpy as np
import numpy.core.numeric as npnum
import copy

from datetime import date
from astropy.io import fits
import matplotlib.pyplot as plt

from pyklip.instruments.Instrument import GenericData
import pyklip.rdi as rdi
import pyklip.instruments.JWST as JWST
import pyklip.parallelized as parallelized
import pyklip.instruments.utils.bkgd as bkgd

from . import io
from . import utils


def perform_subtraction(meta):
    '''
    Perform the PSF subtraction.

    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from
        engine.py.

    '''

    if meta.use_cleaned:
        ext = '_CLEAN'
    else:
        ext = ''

    files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS/SCI+REF'+ext, search=meta.sub_ext)
    # Run some preparation steps on the meta object
    meta = utils.prepare_meta(meta, files)

    if meta.bgsub != 'None':
        print('WARNING: Background subtraction only works if running one filter at a time!')
        bgout = bg_subtraction(meta)
        # Reinitialise meta info for pyKLIP
        files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS/BGSUB', search=meta.sub_ext)
        meta = utils.prepare_meta(meta, files)

    # Perform KLIP subtraction
    if not meta.bgonly:
        klip_subtraction(meta, files)

def bg_subtraction(meta):
    '''
    Perform background subtraction on the processed images
    '''
    if meta.use_cleaned:
        ext = '_CLEAN'
    else:
        ext = ''

    if meta.bgsub in ['default', 'pyklip', 'leastsq']:
        bgsci_files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS/BGSCI'+ext, search=meta.sub_ext)
        bgref_files = io.get_working_files(meta, meta.done_imgprocess, subdir='IMGPROCESS/BGREF'+ext, search=meta.sub_ext)
    elif meta.bgsub == 'saved':
        # Use saved files
        return None
    elif meta.bgsub == 'sci':
        raise ValueError('Not implemented yet!')
    elif meta.bgsub == 'ref':
        raise ValueError('Not implemented yet!')
    else:
        if meta.bgsub == 'None':
            print('Skipping background subtraction')
            return
        else:
            raise ValueError('Background subtraction {} not recognised'.format(meta.bgsub))

    # Trim the first integration if requested.
    if meta.bgtrim == 'first':
        data_start = 1
    else:
        data_start = 0

    types = ['SCI', 'CAL']
    for i, files in enumerate([bgsci_files, bgref_files]):
        
        for key in meta.obs.keys():
            ww = np.where(meta.obs[key]['TYP'] == types[i])[0]
        basefiles = np.array(meta.obs[key]['FITSFILE'][ww], dtype=str).tolist()
        bgmed_split = meta.bgmed_splits[i]

        if meta.bgsub == 'default':
            bg_sub = median_bg(files, data_start=data_start, med_split=bgmed_split)
            #Save the median_bg
            primary = fits.PrimaryHDU(bg_sub)
            hdul = fits.HDUList([primary])
            meddir = meta.ancildir + 'median_bg/'
            if not os.path.exists(meddir):
                os.makedirs(meddir)
            hdul.writeto(meddir+'{}.fits'.format(types[i]), overwrite=meta.overwrite)

            for file in basefiles:
                with fits.open(file) as hdul:
                    data = hdul['SCI'].data[data_start:] 

                    # Split up the array the same way as numpy split the medians
                    data_split = np.array_split(data, bgmed_split, axis=0)
                    # Loop over splits and subtract corresponding median
                    for j in range(bgmed_split):
                        data_split[j] -=  bg_sub[j]
                    # Recombine data
                    hdul['SCI'].data = np.concatenate(data_split,axis=0)

                    if data_start != 0:
                        hdul[0].header['NINTS'] -= data_start 
                        for ext in ['ERR', 'DQ', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:  
                            temp_data = hdul[ext].data
                            hdul[ext].data = np.array(temp_data[data_start:])

                    savedir = '/'.join(file.split('/')[:-2])+'/BGSUB'
                    savefile = file.split('/')[-1].replace('calints', 'bg_calints')
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)
                    hdul.writeto(savedir+'/'+savefile, overwrite=meta.overwrite)
        elif meta.bgsub == 'pyklip':
            klip_bg(basefiles, files, data_start=data_start, overwrite=meta.overwrite)
        elif meta.bgsub == 'leastsq':
            leastsq_bg(meta, basefiles, files, data_start=data_start, overwrite=meta.overwrite)

    return

def median_bg(files, data_start=0, med_split=1):
    '''
    Take a list of bg files and median combine them
    '''

    # Get some information on the array shape
    head = fits.getheader(files[0])
    nints = head['NINTS']
    xdim = head['SUBSIZE1']
    ydim = head['SUBSIZE2']

    # Create array to hold backgrounds
    bg_data = np.empty((len(files)*nints,ydim,xdim))

    # Loop over files and extract background
    for i, file in enumerate(files):
        start = i*nints + data_start
        end = (i+1)*nints

        # Grab the data
        if i != len(files)-1:
            bg_data[start:end] = fits.getdata(file, 'SCI')[data_start:]
        else:
            bg_data[start:] = fits.getdata(file, 'SCI')[data_start:]

    # Take a median of the data
    bg_data = np.array_split(bg_data, med_split, axis=0)
    bg_median = []
    for bgd in bg_data:
        bg_median.append(np.median(bgd, axis=0))
    bg_median = np.stack(bg_median, axis=0)

    return bg_median

def klip_bg(basefiles, bgfiles, data_start=0, overwrite=True):

    data = []
    pas = []
    centers = []
    filenames = []
    wcs_hdrs = []
    for file in basefiles:
        with fits.open(file) as hdu:
            nints = hdu[0].header['NINTS'] - data_start
            filt = hdu[0].header['FILTER']

            #Define data
            img_data = hdu['SCI'].data[data_start:]

            # Define filenames
            filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

            # Define image centers
            if filt == 'F1065C':
                aligned_center = [120.81, 111.89]
            elif filt == 'F1140C':
                aligned_center = [119.99, 112.2]
            elif filt == 'F1550C':
                aligned_center = [119.84, 113.33]
            img_centers = [aligned_center for i in range(nints)]
            # # Get WCS information
            # wcs_hdr = wcs.WCS(header=hdu['SCI'].header, naxis=hdu['SCI'].header['WCSAXES'])
            # for i in range(nints):
            #     wcs_hdrs.append(wcs_hdr.deepcopy())

            # Append to arrays
            data.extend(img_data)
            centers.extend(img_centers) 
            pas.extend([0]*nints)

    dataset = GenericData(data, centers, IWA=0, parangs=np.array(pas), filenames=filenames)

    # Now prep the PSF Library
    psflib_imgs = []
    psflib_filenames = []
    for file in bgfiles:
        with fits.open(file) as hdu:
            nints = hdu[0].header['NINTS'] - data_start
            filt = hdu[0].header['FILTER']

            #Define data
            psfimg_data = hdu['SCI'].data[data_start:]

            # Define filenames
            psflib_filenames += ['{}_INT{}'.format(file.split('/')[-1], i+1) for i in range(nints)]

            # Append to arrays
            psflib_imgs.extend(psfimg_data)

    # # Append the target images as well
    psflib_imgs = np.append(psflib_imgs, dataset._input, axis=0)
    psflib_filenames = np.append(psflib_filenames, dataset._filenames, axis=0)

    psflib = rdi.PSFLibrary(psflib_imgs, aligned_center, psflib_filenames, compute_correlation=True)

    # Now run it
    psflib.prepare_library(dataset)
    numbasis=[20] # number of KL basis vectors to use to model the PSF. We will try several different ones
    maxnumbasis=150 # maximum number of most correlated PSFs to do PCA reconstruction with
    annuli=1
    subsections=1 # break each annulus into 4 sectors
    parallelized.klip_dataset(dataset, outputdir='.', fileprefix="pyklip_k150a3s4m1", annuli=annuli,
                            subsections=subsections, numbasis=numbasis, maxnumbasis=maxnumbasis, mode="RDI",
                            aligned_center=aligned_center, psf_library=psflib, movement=1, save_ints = True)

    klip_sub = dataset.allints[0,0,:,:,:] # Grab all the integrations

    # Now want to save back into the original files so spaceKLIP can understand things
    for i, file in enumerate(basefiles):
        with fits.open(file) as hdu:
            hdu[0].header['NINTS'] -= data_start
            nints = hdu[0].header['NINTS']

            # Swap out data
            if i != len(basefiles)-1:
                hdu['SCI'].data[data_start:] = klip_sub[nints*i:nints*(i+1)]
            else:
                hdu['SCI'].data[data_start:] = klip_sub[nints*i:] 

            # Trim if necessary
            if data_start == 1:
                for ext in ['SCI', 'ERR', 'DQ', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:  
                    temp_data = hdu[ext].data
                    hdu[ext].data = np.array(temp_data[data_start:])

            # Save file
            savedir = '/'.join(file.split('/')[:-2])+'/BGSUB'
            savefile = file.split('/')[-1].replace('calints', 'bg_calints')
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            hdu.writeto(savedir+'/'+savefile, overwrite=overwrite)

    return

def leastsq_bg(meta, basefiles, bgfiles, data_start=0, overwrite=True):
    '''
    Compute background subtraction using a least squares combination of images
    on specific regions of the detector. 
    '''

    # Get filter
    filt = fits.getheader(basefiles[0])['FILTER']

    # Create cube of backgrounds
    bkgd_cube = []
    for file in bgfiles:
        with fits.open(file) as hdu:
            bkgd_cube.append(hdu['SCI'].data[data_start:])
    bkgd_cube = np.concatenate(bkgd_cube)

    if filt == 'F1065C':
        mask_x, mask_y = 120.81, 111.89
    elif filt == 'F1140C':
        mask_x, mask_y = 119.99, 112.2
    elif filt == 'F1550C':
        mask_x, mask_y = 119.84, 113.33
        mask_str = '/../resources/miri_transmissions/jwst_miri_psfmask_1550_jasonrotate.fits'
        maskfile = os.path.join(os.path.dirname(__file__) + mask_str)
        bmask_str = '/../resources/miri_transmissions/jwst_miri_psfmask_1550_raw_extract.fits'
        bmaskfile = os.path.join(os.path.dirname(__file__) + bmask_str)

    # Get mask
    with fits.open(maskfile) as hdu:
        full_mask = hdu['SCI'].data

    # Get blank mask
    with fits.open(bmaskfile) as hdu:
        blank = hdu['SCI'].data

    y, x = np.indices(blank.shape)
    r = np.sqrt((x - mask_x)**2 + (y - mask_y)**2)

    # Create glowstick subtraction
    glowstick_sub = np.copy(blank)
    glowstick_sub[np.where((full_mask > 0.5))] = 0 # this basically just selects the glowstick
    glowstick_opt = np.copy(glowstick_sub) 
    glowstick_opt[np.where(r < 40)] = 0 # the optimizaiton region should not include the mask center that contains the star

    ## create the mask for the rest of the thermal background
    therm_sub = np.copy(blank)
    therm_sub[np.where((glowstick_sub == 1) & (blank == 1)) ] = 0 # mask out valid regions that the glowstick mask is using
    therm_opt = np.copy(therm_sub)
    therm_opt[np.where(r < 40)] = 0 # mask out central star

    glowstick_sub[140:,:] = 0
    glowstick_sub[:75,:] = 0
    glowstick_opt[140:,:] = 0
    glowstick_opt[:75,:] = 0

    # fig = plt.figure(figsize=(7,4))
    # fig.add_subplot(1,2,1)
    # plt.imshow(glowstick_sub)
    # plt.title("Subtraction zone")

    # fig.add_subplot(1,2,2)
    # plt.imshow(glowstick_opt)
    # plt.title("Optimization zone")
    # plt.show()

    # fig = plt.figure(figsize=(7,4))
    # fig.add_subplot(1,2,1)
    # plt.imshow(therm_sub)
    # plt.title("Subtraction zone")

    # fig.add_subplot(1,2,2)
    # plt.imshow(therm_opt)
    # plt.title("Optimization zone")
    # plt.show()
    # exit()

    bkgd_cube[np.where(np.isnan(bkgd_cube))] = 0

    for file in basefiles:
        with fits.open(file) as hdu:
            hdu[0].header['NINTS'] -= data_start
            sci_cube = hdu['SCI'].data[data_start:]
            bkgdsub_cube = bkgd.subtract_bkgd(sci_cube, bkgd_cube, [therm_opt, glowstick_opt], [therm_sub, glowstick_sub], conserve_flux=True)

            bkgdsub_cube = np.nan_to_num(bkgdsub_cube, 0)
            hdu['SCI'].data = bkgdsub_cube
            
            # Trim if necessary
            if data_start == 1:
                for ext in ['ERR', 'DQ', 'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT']:  
                    temp_data = hdu[ext].data
                    hdu[ext].data = np.array(temp_data[data_start:])

            # Save file
            savedir = '/'.join(file.split('/')[:-2])+'/BGSUB'
            savefile = file.split('/')[-1].replace('calints', 'bg_calints')
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            hdu.writeto(savedir+'/'+savefile, overwrite=overwrite)

    return


def klip_subtraction(meta, files):
    """
    Run pyKLIP.

    Parameters
    ----------
    meta : class
        Meta class containing data and configuration information from
        engine.py.
    """

    if meta.verbose:
        print('--> Running pyKLIP...')

    # Loop through all modes, numbers of annuli, and numbers of subsections.
    Nscenarios = len(meta.mode)*len(meta.annuli)*len(meta.subsections)
    counter = 1
    meta.truenumbasis = {}
    meta.rundirs = [] # create an array to save the run directories to
    for mode in meta.mode:
        for annuli in meta.annuli:
            for subsections in meta.subsections:

                # Update terminal if requested
                if meta.verbose:
                    print('--> Mode = '+mode+', annuli = %.0f, subsections = %.0f, scenario %.0f of %.0f' % (annuli, subsections, counter, Nscenarios))

                # Create an output directory for each set of pyKLIP parameters
                today = date.today().strftime('%Y_%m_%d_')
                odir = meta.odir+today+mode+'_annu{}_subs{}_run'.format(annuli, subsections)

                # Figure out how many runs of this type have already been
                # performed
                existing_runs = glob.glob(odir+'*'.format(today))

                # Assign run number based on existing runs
                odir += str(len(existing_runs)+1)+'/'

                # Save the odir to the meta object for later analyses
                meta.rundirs.append(odir)

                # Now provide and create actual directory to save to
                odir += 'SUBTRACTED/'
                if not os.path.exists(odir):
                    os.makedirs(odir)
                if not os.path.exists(meta.ancildir+'shifts'):
                    os.makedirs(meta.ancildir+'shifts')

                # Loop through all sets of observing parameters. Only run
                # pyKLIP if the corresponding KLmodes-all fits file does
                # not exist yet.
                for i, key in enumerate(meta.obs.keys()):
                    meta.truenumbasis[key] = [num for num in meta.numbasis if (num <= meta.maxnumbasis[key])]
                    if meta.overwrite == False and os.path.exists(odir+'-KLmodes-all.fits'):
                        continue
                    ww_sci = np.where(meta.obs[key]['TYP'] == 'SCI')[0]
                    filepaths = np.array(meta.obs[key]['FITSFILE'][ww_sci], dtype=str).tolist()
                    ww_cal = np.where(meta.obs[key]['TYP'] == 'CAL')[0]
                    psflib_filepaths = np.array(meta.obs[key]['FITSFILE'][ww_cal], dtype=str).tolist()
                    dataset = JWST.JWSTData(filepaths=filepaths,
                                            psflib_filepaths=psflib_filepaths, centering=meta.centering_alg, badpix_threshold=meta.badpix_threshold,
                                            scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                            fiducial_point_override=meta.fiducial_point_override)
                    
                    #Set an OWA if it exists. 
                    if hasattr(meta, 'OWA'): dataset.OWA = meta.OWA

                    #If algo is not set assume klip
                    if hasattr(meta, 'algo'): algo = meta.algo else: algo='klip'
                    parallelized.klip_dataset(dataset=dataset,
                                              mode=mode,
                                              outputdir=odir,
                                              fileprefix=key,
                                              annuli=annuli,
                                              subsections=subsections,
                                              movement=1,
                                              numbasis=meta.truenumbasis[key],
                                              calibrate_flux=False,
                                              maxnumbasis=meta.maxnumbasis[key],
                                              psf_library=dataset.psflib,
                                              highpass=False,
                                              verbose=meta.verbose,
                                              algo=meta.algorithm)

                # Save a meta file under each directory
                smeta = copy.deepcopy(meta)
                smeta.used_mode = mode
                smeta.used_annuli = annuli
                smeta.used_subsections = subsections

                io.meta_to_json(smeta, savefile=odir+'MetaSave.json')

                # Increment counter
                counter += 1
    
    return
