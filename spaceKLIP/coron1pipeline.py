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

from tqdm.auto import trange

from jwst.lib import reffile_utils
from jwst.datamodels import dqflags, RampModel, SaturationModel
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline

from webbpsf_ext import robust

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


# =============================================================================
# MAIN
# =============================================================================

class Coron1Pipeline_spaceKLIP(Detector1Pipeline):
    """
    The spaceKLIP JWST stage 1 pipeline class.
    
    Apply all calibration steps to raw JWST ramps to produce 
    a 2-D slope product. Custom sub-class of ``Detector1Pipeline`` 
    with modifications for coronagraphic data.
    
    Included steps are: group_scale, dq_init, saturation, ipc, 
    superbias, refpix, rscd, firstframe, lastframe, linearity, 
    dark_current, reset, persistence, jump detection, ramp_fit, 
    and gain_scale. 
    """
    
    class_alias = "calwebb_coron1"

    spec = """
        remove_ktc         = boolean(default=True) # Remove kTC noise from data
        remove_fnoise      = boolean(default=True) # Remove 1/f noise from data
    """
    
    def __init__(self,
                 **kwargs):
        """
        Initialize the spaceKLIP JWST stage 1 pipeline class.
        
        Parameters
        ----------
        **kwargs : keyword arguments
            Default JWST stage 1 detector pipeline keyword arguments.
        
        Returns
        -------
        None.
        
        """
        
        # Initialize Detector1Pipeline class.
        super(Coron1Pipeline_spaceKLIP, self).__init__(**kwargs)
        
        # Set additional step parameters.
        self.saturation.grow_diagonal = False
        self.refpix.nlower = 4
        self.refpix.nupper = 4
        self.refpix.nleft = 4
        self.refpix.nright = 4
        self.refpix.nrow_off = 0
        self.refpix.ncol_off = 0
        self.ramp_fit.save_calibrated_ramp = False
        
        pass
    
    def process(self,
                input):
        """
        Process an input JWST datamodel with the spaceKLIP JWST stage 1
        pipeline.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        
        Returns
        -------
        output : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Open input as ramp model.
        input = RampModel(input)
        
        # Process MIR & NIR exposures differently.
        instrument = input.meta.instrument.name
        if instrument == 'MIRI':
            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.run_step(self.saturation, input)
            input = self.run_step(self.ipc, input)
            input = self.run_step(self.firstframe, input)
            input = self.run_step(self.lastframe, input)
            input = self.run_step(self.reset, input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.rscd, input)
            input = self.run_step(self.dark_current, input)
            input = self.do_refpix(input)
            input = self.run_step(self.jump, input)
        else:
            input = self.run_step(self.group_scale, input)
            input = self.run_step(self.dq_init, input)
            input = self.do_saturation(input)
            input = self.run_step(self.ipc, input)
            input = self.run_step(self.superbias, input)
            input = self.do_refpix(input)
            input = self.run_step(self.linearity, input)
            input = self.run_step(self.persistence, input)
            input = self.run_step(self.dark_current, input)
            input = self.run_step(self.jump, input)
            if self.remove_ktc or self.remove_fnoise:
                input = self.subtract_ktc(input)
            if self.remove_fnoise:
                input = self.subtract_fnoise(input, model_type='savgol')
        
        # Save calibrated ramp data.
        if self.ramp_fit.save_calibrated_ramp:
            self.save_model(input, suffix='ramp')
        
        # Run ramp fitting & gain scale correction.
        res = self.run_step(self.ramp_fit, input, save_results=False)
        input, ints_model = (res, None) if self.ramp_fit.skip else res
        if input is None:
            self.ramp_fit.log.info('NoneType returned from ramp fitting. Gain scale correction skipped')
        else:
            self.gain_scale.suffix = 'rate'
            input = self.run_step(self.gain_scale, input, save_results=False)
        if ints_model is not None:
            self.gain_scale.suffix = 'rateints'
            ints_model = self.run_step(self.gain_scale, ints_model, save_results=False)
            if self.save_results:
                self.save_model(ints_model, suffix='rateints')
        
        # Setup output file.
        self.setup_output(input)
        
        return input
    
    def run_step(self,
                 step_obj,
                 input,
                 save_results=None,
                 **kwargs):
        """
        Run a JWST pipeline step.
        
        Parameters
        ----------
        step_obj : jwst.step
            JWST pipeline step to be run.
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        save_results : bool, optional
            Save the JWST pipeline step product? None will default to the JWST
            pipeline step default. The default is None.
        **kwargs : keyword arguments
            Default JWST pipeline step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Check if results shall be saved.
        if step_obj.skip:
            really_save_results = False
        elif save_results is not None:
            really_save_results = save_results
        elif step_obj.save_results:
            really_save_results = True
        else:
            really_save_results = False
        
        # Run step. Don't save results yet.
        step_save_orig = step_obj.save_results
        step_obj.save_results = False
        res = step_obj(input)
        step_obj.save_results = step_save_orig
        
        # Check if group scale correction or gain scale correction were
        # skipped.
        if step_obj is self.group_scale:
            if res.meta.cal_step.group_scale == 'SKIPPED':
                really_save_results = False
        elif step_obj is self.gain_scale:
            if res.meta.cal_step.gain_scale == 'SKIPPED':
                really_save_results = False
        
        # Save results.
        if really_save_results:
            step_obj.output_dir = self.output_dir
            if isinstance(res, (tuple)):
                self.save_model(res[0], suffix=step_obj.suffix+'0')
                self.save_model(res[1], suffix=step_obj.suffix+'1')
            else:
                self.save_model(res, suffix=step_obj.suffix)
        
        return res
    
    def do_saturation(self,
                      input,
                      **kwargs):
        """
        Do the default or a custom saturation correction.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 saturation step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Save original step parameter.
        npix_grow = self.saturation.n_pix_grow_sat
        
        # Run step with default settings.
        if self.saturation.grow_diagonal or npix_grow == 0:
            return self.run_step(self.saturation, input, **kwargs)
        
        # Run step with 1 fewer growth.
        self.saturation.n_pix_grow_sat = npix_grow - 1
        res = self.run_step(self.saturation, input, **kwargs)
        
        # Get saturated pixels and flag 4 neighbors.
        mask_sat = res.groupdq & dqflags.pixel['SATURATED'] == dqflags.pixel['SATURATED']
        mask_vp1 = np.roll(mask_sat, +1, axis=-2)
        mask_vm1 = np.roll(mask_sat, -1, axis=-2)
        mask_hp1 = np.roll(mask_sat, +1, axis=-1)
        mask_hm1 = np.roll(mask_sat, -1, axis=-1)
        
        # Ignore pixels which rolled around array border.
        mask_vp1[:, :,  0, :] = 0
        mask_vm1[:, :, -1, :] = 0
        mask_hp1[:, :, :,  0] = 0
        mask_hm1[:, :, :, -1] = 0
        
        # Combine saturation masks.
        mask_sat = mask_sat | mask_vp1 | mask_vm1 | mask_hp1 | mask_hm1
        
        # Flag saturated pixels.
        res.groupdq = res.groupdq | (mask_sat * dqflags.pixel['SATURATED'])
        
        # Delete unused masks.
        del mask_vp1, mask_vm1, mask_hp1, mask_hm1
        
        return res
    
    def do_refpix(self,
                  input,
                  **kwargs):
        """
        Do the default or a custom pseudo reference pixel correction.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 refpix step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Check if full frame exposure.
        is_full_frame = 'FULL' in input.meta.subarray.name.upper()
        
        # Get number of custom reference pixel rows & columns.
        nlower = self.refpix.nlower
        nupper = self.refpix.nupper
        nleft = self.refpix.nleft
        nright = self.refpix.nright
        nall = nlower + nupper + nleft + nright
        
        # Run step with default settings if full frame exposure, otherwise run
        # step with custom reference pixel rows & columns.
        if is_full_frame or nall == 0:
            return self.run_step(self.refpix, input, **kwargs)
        else:
            return self.do_pseudo_refpix(input, **kwargs)
    
    def do_pseudo_refpix(self,
                         input,
                         **kwargs):
        """
        Do a pseudo reference pixel correction. Therefore, flag the requested
        edge rows and columns as reference pixels, run the JWST stage 1 refpix
        step, and unflag the pseudo reference pixels again.
        
        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.
        **kwargs : keyword arguments
            Default JWST stage 1 refpix step keyword arguments.
        
        Returns
        -------
        res : jwst.datamodel
            Output JWST datamodel.
        
        """
        
        # Get number of custom reference pixel rows & columns.
        nlower = self.refpix.nlower
        nupper = self.refpix.nupper
        nleft = self.refpix.nleft
        nright = self.refpix.nright
        nrow_off = self.refpix.nrow_off
        ncol_off = self.refpix.ncol_off
        nupper_off = -nrow_off if nrow_off != 0 else None
        nright_off = -ncol_off if ncol_off != 0 else None
        
        # Flag custom reference pixel rows & columns.
        self.refpix.log.info(f'Flagging [{nlower}, {nupper}] references rows at [bottom, top] of array')
        self.refpix.log.info(f'Flagging [{nleft}, {nright}] references columns at [left, right] of array')
        input.pixeldq[nrow_off:nrow_off + nlower, ncol_off:nright_off] = input.pixeldq[nrow_off:nrow_off + nlower, ncol_off:nright_off] | dqflags.pixel['REFERENCE_PIXEL']
        input.pixeldq[-nrow_off - nupper:nupper_off, ncol_off:nright_off] = input.pixeldq[-nrow_off - nupper:nupper_off, ncol_off:nright_off] | dqflags.pixel['REFERENCE_PIXEL']
        input.pixeldq[nrow_off:nupper_off, ncol_off:ncol_off + nleft] = input.pixeldq[nrow_off:nupper_off, ncol_off:ncol_off + nleft] | dqflags.pixel['REFERENCE_PIXEL']
        input.pixeldq[nrow_off:nupper_off, -ncol_off - nright:nright_off] = input.pixeldq[nrow_off:nupper_off, -ncol_off - nright:nright_off] | dqflags.pixel['REFERENCE_PIXEL']
        
        # Save original step parameter.
        use_side_orig = self.refpix.use_side_ref_pixels
        if nleft + nright == 0:
            self.refpix.use_side_ref_pixels = False
        else:
            self.refpix.use_side_ref_pixels = True
        
        # Run step with custom reference pixel rows & columns.
        res = self.run_step(self.refpix, input, **kwargs)
        
        # Apply original step parameter.
        self.refpix.use_side_ref_pixels = use_side_orig
        
        # Unflag custom reference pixel rows & columns.
        self.refpix.log.info('Removing custom reference pixel flags')
        res.pixeldq[nrow_off:nrow_off + nlower, ncol_off:nright_off] = res.pixeldq[nrow_off:nrow_off + nlower, ncol_off:nright_off] & ~dqflags.pixel['REFERENCE_PIXEL']
        res.pixeldq[-nrow_off - nupper:nupper_off, ncol_off:nright_off] = res.pixeldq[-nrow_off - nupper:nupper_off, ncol_off:nright_off] & ~dqflags.pixel['REFERENCE_PIXEL']
        res.pixeldq[nrow_off:nupper_off, ncol_off:ncol_off + nleft] = res.pixeldq[nrow_off:nupper_off, ncol_off:ncol_off + nleft] & ~dqflags.pixel['REFERENCE_PIXEL']
        res.pixeldq[nrow_off:nupper_off, -ncol_off - nright:nright_off] = res.pixeldq[nrow_off:nupper_off, -ncol_off - nright:nright_off] & ~dqflags.pixel['REFERENCE_PIXEL']
        
        return res

    def _fit_slopes(self,
                    input,
                    sat_frac=0.5):
        """Fit slopes to each integration
        
        Uses custom `cube_fit` function to fit slopes to each integration.
        Returns aray of slopes and bias values for each integration.
        Bias and slope arrays have shape (nints, ny, nx).

        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel housing the data to be fit.
        sat_frac : float
            Saturation threshold for fitting. Values above
            this fraction of the saturation level are ignored.
            Default is 0.5 to ensure that the fit is within 
            the linear range.
        """

        from .imagetools import cube_fit

        # Get saturation reference file
        # Get the name of the saturation reference file
        sat_name = self.saturation.get_reference_file(input, 'saturation')

        # Open the reference file data model
        sat_model = SaturationModel(sat_name)

        # Extract subarray from saturation reference file, if necessary
        if reffile_utils.ref_matches_sci(input, sat_model):
            sat_thresh = sat_model.data.copy()
        else:
            ref_sub_model = reffile_utils.get_subarray_model(input, sat_model)
            sat_thresh = ref_sub_model.data.copy()
            ref_sub_model.close()

        # Close the reference file
        sat_model.close()

        # Perform ramp fit to data to get bias offset
        group_time = input.meta.exposure.group_time
        ngroups = input.meta.exposure.ngroups
        nints = input.meta.exposure.nints
        tarr = np.arange(1, ngroups+1) * group_time
        data = input.data

        bias_arr = []
        slope_arr = []
        for i in range(nints):
            # Get group-level bpmask for this integration
            groupdq = input.groupdq[i]
            # Make sure to accumulate the group-level dq mask
            bpmask_arr = np.cumsum(groupdq, axis=0) > 0
            cf = cube_fit(tarr, data[i], bpmask_arr=bpmask_arr,
                          sat_vals=sat_thresh, sat_frac=sat_frac)
            bias_arr.append(cf[0])
            slope_arr.append(cf[1])
        bias_arr = np.asarray(bias_arr)
        slope_arr = np.asarray(slope_arr)

        # bias and slope arrays have shape [nints, ny, nx]
        # bias values are in units of DN and slope in DN/sec
        return bias_arr, slope_arr

    def subtract_ktc(self,
                     input,
                     sat_frac=0.5):
        
        bias_arr, _ = self._fit_slopes(input, sat_frac=sat_frac)

        # Subtract bias from each integration
        nints = input.meta.exposure.nints
        for i in range(nints):
            input.data[i] -= bias_arr[i]

        return input
    
    def subtract_fnoise(self,
                        input,
                        sat_frac=0.5,
                        **kwargs):
        """Model and subtract 1/f noise from each integration
        
        TODO: Make this into a Step class.
        TODO: Automatic function to determine if correction is necessary.

        Parameters
        ----------
        input : jwst.datamodel
            Input JWST datamodel to be processed.

        Keyword Args
        ------------
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        """
        
        from .fnoise_clean import CleanSubarray

        is_full_frame = 'FULL' in input.meta.subarray.name.upper()
        nints    = input.meta.exposure.nints
        ngroups  = input.meta.exposure.ngroups
        noutputs = input.meta.exposure.noutputs

        if is_full_frame:
            assert noutputs == 4, 'Full frame data must have 4 outputs'
        else:
            assert noutputs == 1, 'Subarray data must have 1 output'

        ny, nx = input.data.shape[-2:]
        chsize = ny // noutputs

        # Fit slopes to get signal mask
        # Grab slopes if they've already been computed
        _, slope_arr = self._fit_slopes(input, sat_frac=sat_frac)
        slope_mean = robust.mean(slope_arr, axis=0)

        # Generate a mean signal ramp to subtract from each group
        group_time = input.meta.exposure.group_time
        ngroups = input.meta.exposure.ngroups
        tarr = np.arange(1, ngroups+1) * group_time
        signal_mean_ramp = slope_mean * tarr.reshape([-1,1,1])

        # Subtract 1/f noise from each integration
        data = input.data
        for i in trange(nints):
            cube = data[i]
            groupdq = input.groupdq[i]
            # Cumulative sum of group DQ flags
            bpmask_arr = np.cumsum(groupdq, axis=0) > 0
            for j in range(ngroups):
                # Exclude bad pixels
                im_mask = ~bpmask_arr[j] #& mask
                for ch in range(noutputs):
                    # Get channel x-indices
                    x1 = int(ch*chsize)
                    x2 = int(x1 + chsize)

                    # Channel subarrays
                    imch = cube[j, :, x1:x2]
                    sigch = signal_mean_ramp[j, :, x1:x2]
                    good_mask = im_mask[:, x1:x2]

                    # Remove averaged signal goup
                    imch_diff = imch - sigch

                    # Subtract 1/f noise
                    nf_clean = CleanSubarray(imch_diff, good_mask)
                    nf_clean.fit(**kwargs)
                    # Subtract model from data
                    data[i,j,:,x1:x2] -= nf_clean.model

                    del nf_clean

        return input

def run_obs(database,
            steps={},
            subdir='stage1'):
    """
    Run the JWST stage 1 detector pipeline on the input observations database.
    This customized implementation can:
    - Do a custom saturation correction where only the bottom/top/left/right
      and not the diagonal pixels next to a saturated pixel are flagged.
    - Do a pseudo reference pixel correction. Therefore, flag the requested
      edge rows and columns as reference pixels, run the JWST stage 1 refpix
      step, and unflag the pseudo reference pixels again.
    
    Parameters
    ----------
    database : spaceKLIP.Database
        SpaceKLIP database on which the JWST stage 1 detector pipeline shall be
        run.
    steps : dict, optional
        See here for how to use the steps parameter:
        https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/running_pipeline_python.html#configuring-a-pipeline-step-in-python
        Custom step parameters are:
        - saturation/grow_diagonal : bool, optional
            Flag also diagonal pixels (or only bottom/top/left/right)? The
            default is True.
        - refpix/nlower : int, optional
            Number of rows at frame bottom that shall be used as additional
            reference pixels. The default is 0.
        - refpix/nupper : int, optional
            Number of rows at frame top that shall be used as additional
            reference pixels. The default is 0.
        - refpix/nleft : int, optional
            Number of rows at frame left side that shall be used as additional
            reference pixels. The default is 0.
        - refpix/nright : int, optional
            Number of rows at frame right side that shall be used as additional
            reference pixels. The default is 0.
        - ramp_fit/save_calibrated_ramp : bool, optional
            Save the calibrated ramp? The default is False.
        The default is {}.
    subdir : str, optional
        Name of the directory where the data products shall be saved. The
        default is 'stage1'.
    
    Returns
    -------
    None.
    
    """
    
    # Set output directory.
    output_dir = os.path.join(database.output_dir, subdir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through concatenations.
    for i, key in enumerate(database.obs.keys()):
        log.info('--> Concatenation ' + key)
        
        # Loop through FITS files.
        nfitsfiles = len(database.obs[key])
        for j in range(nfitsfiles):
            
            # Skip non-stage 0 files.
            head, tail = os.path.split(database.obs[key]['FITSFILE'][j])
            if database.obs[key]['DATAMODL'][j] != 'STAGE0':
                log.info('  --> Coron1Pipeline: skipping non-stage 0 file ' + tail)
                continue
            log.info('  --> Coron1Pipeline: processing ' + tail)
            
            # Initialize Coron1Pipeline.
            pipeline = Coron1Pipeline_spaceKLIP(output_dir=output_dir)
            pipeline.save_results = True
            
            # Set step parameters.
            for key1 in steps.keys():
                for key2 in steps[key1].keys():
                    setattr(getattr(pipeline, key1), key2, steps[key1][key2])
            
            # Run Coron1Pipeline.
            fitspath = os.path.abspath(database.obs[key]['FITSFILE'][j])
            res = pipeline.run(fitspath)
            if isinstance(res, list):
                res = res[0]
            
            # Update spaceKLIP database.
            fitsfile = os.path.join(output_dir, res.meta.filename)
            if fitsfile.endswith('rate.fits'):
                if os.path.isfile(fitsfile.replace('rate.fits', 'rateints.fits')):
                    fitsfile = fitsfile.replace('rate.fits', 'rateints.fits')
            database.update_obs(key, j, fitsfile)
    
    pass
