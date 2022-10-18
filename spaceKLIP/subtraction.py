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

# Define logging
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

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
        log.warning('Background subtraction only works if running one filter at a time!')
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
            log.info('Skipping background subtraction')
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
        mask_str = '/resources/miri_transmissions/jwst_miri_psfmask_1550_jasonrotate.fits'
        maskfile = os.path.join(os.path.dirname(os.path.abspath(__file__)) + mask_str)
        bmask_str = '/resources/miri_transmissions/jwst_miri_psfmask_1550_raw_extract.fits'
        bmaskfile = os.path.join(os.path.dirname(os.path.abspath(__file__)) + bmask_str)

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
        log.info('--> Running pyKLIP...')

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
                    log.info(f'--> Mode = {mode}, annuli = {annuli:.0f}, subsections = {subsections:.0f}, scenario {counter:.0f} of {Nscenarios:.0f}')

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

                if not hasattr(meta, 'blur_images'):
                    meta.blur_images = False

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
                    load_file0_center = meta.load_file0_center if hasattr(meta,'load_file0_center')  else False

                    dataset = JWST.JWSTData(filepaths=filepaths, psflib_filepaths=psflib_filepaths, centering=meta.centering_alg, 
                                            scishiftfile=meta.ancildir+'shifts/scishifts', refshiftfile=meta.ancildir+'shifts/refshifts',
                                            fiducial_point_override=meta.fiducial_point_override, blur=meta.blur_images,
                                            load_file0_center=load_file0_center,save_center_file=meta.ancildir+'shifts/file0_centers',
                                            spectral_type=meta.spt)
                    #Set an OWA if it exists. 
                    if hasattr(meta, 'OWA'): dataset.OWA = meta.OWA

                    #If algo is not set assume klip
                    algo = meta.algorithm if hasattr(meta, 'algorithm') else 'klip'
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
                                              algo=algo)

                # Save a meta file under each directory
                smeta = copy.deepcopy(meta)
                smeta.used_mode = mode
                smeta.used_annuli = annuli
                smeta.used_subsections = subsections

                io.meta_to_json(smeta, savefile=odir+'MetaSave.json')

                # Increment counter
                counter += 1
    
    return


###############################
#### JWST Pipeline updates ####
####   UNDER DEVELOPMENT   ####
###############################

from jwst import datamodels
from jwst.datamodels import dqflags
from jwst.pipeline import Coron3Pipeline
from jwst.model_blender import blendmeta
from collections import defaultdict

from jwst.coron.align_refs_step import AlignRefsStep, median_replace_img, imageregistration

class AlignRefsStepUpdates(AlignRefsStep):

    """
    AlignRefsStepUpdates: Align coronagraphic PSF images
    with science target images, excluding mask transmission
    """

    class_alias = "align_refs"

    spec = """
        exclude_psfmask = boolean(default=True) # Don't weight alignment by coron image mask
        median_replace = boolean(default=True)  # Replace bad pixels with median?
    """

    reference_file_types = ['psfmask']

    def process(self, target, psf):

        # Open the input science target model
        with datamodels.open(target) as target_model:

            # Get the name of the psf mask reference file to use
            self.mask_name = self.get_reference_file(target_model, 'psfmask')
            self.log.info('Using PSFMASK reference file %s', self.mask_name)

            # Check for a valid reference file
            if self.mask_name == 'N/A':
                self.log.warning('No PSFMASK reference file found')
                self.log.warning('Align_refs step will be skipped')
                return None

            # Open the psf mask reference file
            mask_model = datamodels.ImageModel(self.mask_name)
            # Update all pixels to weighting=1
            if self.exclude_psfmask:
                mask_model.data[:] = 1

            # Open the input psf images
            psf_model = datamodels.open(psf)

            # Retrieve the box size for the filter
            box_size = self.median_box_length

            # Get the bit value of bad pixels. A value of 0 treats all pixels as good.
            bad_bitvalue = self.bad_bits
            bad_bitvalue = dqflags.interpret_bit_flags(bad_bitvalue, mnemonic_map=dqflags.pixel)
            if bad_bitvalue is None:
                bad_bitvalue = 0

            # Replace bad pixels in the target and psf images?
            if self.median_replace:
                target_model = median_replace_img(target_model, box_size, bad_bitvalue)
                psf_model = median_replace_img(psf_model, box_size, bad_bitvalue)

            # Call the alignment routine
            result = imageregistration.align_models(target_model, psf_model,
                                                    mask_model)
            result.meta.cal_step.align_psfs = 'COMPLETE'

            mask_model.close()
            psf_model.close()

        return result

class Coron3PipelinePYKLIP(Coron3Pipeline):
    """Class for defining Coron3PipelinePYKLIP.

    Modified to include:
        - pixel cleaning algorithm
        - pyKLIP PSF subtraction infrastructure (TBD)

    Coron3PipelineSPYKLIP: Apply all level-3 calibration steps to a
    coronagraphic association of exposures. Included steps are:

    #. stack_refs (assemble reference PSF inputs)
    #. outlier_detection (flag outliers)
    #. align_refs (align reference PSFs to target images)
    #. klip (PSF subtraction using the KLIP algorithm)
    #. resample (image combination and resampling)

    """

    class_alias = "calwebb_coron3_pyklip"

    spec = """
        clean_data = boolean(default=True)
    """

    def __init__(self, *args, **kwargs):

        # Replace align_refs step with different Step class
        steps_update = {
            'align_refs': AlignRefsStepUpdates,
        }
        for k in steps_update.keys():
            self.step_defs[k] = steps_update[k]

        Coron3Pipeline.__init__(self, *args, **kwargs)

    # Main processing
    def process(self, user_input):
        """Primary method for performing pipeline.

        Parameters
        ----------
        user_input : str, Level3 Association, or ~jwst.datamodels.DataModel
            The exposure or association of exposures to process
        """
        self.log.info(f'Starting {self.class_alias} ...')
        asn_exptypes = ['science', 'psf']

        # Create a DM object using the association table
        input_models = datamodels.open(user_input, asn_exptypes=asn_exptypes)
        acid = input_models.meta.asn_table.asn_id

        # Store the output file for future use
        self.output_file = input_models.meta.asn_table.products[0].name

        # Find all the member types in the product
        members_by_type = defaultdict(list)
        prod = input_models.meta.asn_table.products[0].instance

        for member in prod['members']:
            members_by_type[member['exptype'].lower()].append(member['expname'])

        # Set up required output products and formats
        self.outlier_detection.suffix = f'{acid}_crfints'
        self.outlier_detection.save_results = self.save_results
        self.resample.blendheaders = False

        # Save the original outlier_detection.skip setting from the
        # input, because it may get toggled off within loops for
        # processing individual inputs
        skip_outlier_detection = self.outlier_detection.skip

        # Extract lists of all the PSF and science target members
        psf_files = members_by_type['psf']
        targ_files = members_by_type['science']

        # Make sure we found some PSF and target members
        if len(psf_files) == 0:
            err_str1 = 'No reference PSF members found in association table.'
            self.log.error(err_str1)
            self.log.error('Calwebb_coron3 processing will be aborted')
            return

        if len(targ_files) == 0:
            err_str1 = 'No science target members found in association table'
            self.log.error(err_str1)
            self.log.error('Calwebb_coron3 processing will be aborted')
            return

        for member in psf_files + targ_files:
            self.prefetch(member)

        # Assemble all the input psf files into a single ModelContainer
        psf_models = datamodels.ModelContainer()
        for i in range(len(psf_files)):
            psf_input = datamodels.CubeModel(psf_files[i])
            psf_models.append(psf_input)

            psf_input.close()

        # Perform outlier detection on the PSFs.
        if not skip_outlier_detection:
            for model in psf_models:
                self.outlier_detection(model)
                # step may have been skipped for this model;
                # turn back on for next model
                self.outlier_detection.skip = False
        else:
            self.log.info('Outlier detection skipped for PSF\'s')
            
        #### Clean model data in place
        if self.clean_data:
            for model in psf_models:
                model.data = self.clean_images(model)

        # Stack all the PSF images into a single CubeModel
        psf_stack = self.stack_refs(psf_models)
        psf_models.close()

        # Save the resulting PSF stack
        self.save_model(psf_stack, suffix='psfstack')

        # Call the sequence of steps: outlier_detection, align_refs, and klip
        # once for each input target exposure
        resample_input = datamodels.ModelContainer()
        for target_file in targ_files:
            with datamodels.open(target_file) as target:

                # Remove outliers from the target
                if not skip_outlier_detection:
                    target = self.outlier_detection(target)
                    # step may have been skipped for this model;
                    # turn back on for next model
                    self.outlier_detection.skip = False
                    
                #### Clean science data in place
                median_replace_orig = self.align_refs.median_replace
                if self.clean_data:
                    target.data = self.clean_images(target)
                    # No need to replace bad pixels in align_refs Step
                    self.align_refs.median_replace = False

                # Call align_refs
                psf_aligned = self.align_refs(target, psf_stack)
                self.align_refs.median_replace = median_replace_orig

                # Save the alignment results
                self.save_model(
                    psf_aligned, output_file=target_file,
                    suffix='psfalign', acid=acid
                )

                # Call KLIP
                psf_sub = self.do_klip_step(target, psf_aligned)
                psf_sub = self.klip(target, psf_aligned)
                psf_aligned.close()

                # Save the psf subtraction results
                self.save_model(
                    psf_sub, output_file=target_file,
                    suffix='psfsub', acid=acid
                )
                
                #### Clean PSF-subtracted data
                if self.clean_data:
                    psf_sub.data = self.clean_images(psf_sub)

                    # Save the psf subtraction results
                    self.save_model(
                        psf_sub, output_file=target_file,
                        suffix='psfsub_cleaned', acid=acid
                    )

                # Split out the integrations into separate models
                # in a ModelContainer to pass to `resample`
                for model in psf_sub.to_container():
                    resample_input.append(model)

        # Call the resample step to combine all psf-subtracted target images
        result = self.resample(resample_input)

        # Blend the science headers
        try:
            completed = result.meta.cal_step.resample
        except AttributeError:
            self.log.debug('Could not determine if resample was completed.')
            self.log.debug('Presuming not.')

            completed = 'SKIPPED'
        if completed == 'COMPLETE':
            self.log.debug(f'Blending metadata for {result}')
            blendmeta.blendmodels(result, inputs=targ_files)

        try:
            result.meta.asn.pool_name = input_models.meta.asn_table.asn_pool
            result.meta.asn.table_name = os.path.basename(user_input)
        except AttributeError:
            self.log.debug('Cannot set association information on final')
            self.log.debug(f'result {result}')

        # Save the final result
        self.save_model(result, suffix=self.suffix)

        # We're done
        self.log.info(f'...ending {self.class_alias}')

        return

    def clean_images(self, model, **kwargs):
        """Perform bad pixel fixing on flagged and outlier pixels."""
        return utils.clean_data(model.data, model.dq, in_place=True, **kwargs)

    def do_klip_step(self, target, psfs_aligned, kl_modes=None):
        """Perform KLIP subtraction
        
        Includes option to specify number of KL modes.
        """

        # Just do default
        if kl_modes is None:
            return self.klip(target, psfs_aligned)
        else:
            kl_orig = self.klip.truncate
            self.klip.truncate = kl_modes
            
            # Perform KLIP subtraction
            psf_sub = self.klip(target, psfs_aligned)

            # Return to original value
            self.klip.truncate = kl_orig

            return psf_sub


