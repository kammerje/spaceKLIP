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

from jwst.datamodels import dqflags, RampModel
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Coron3Pipeline

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
        save_intermediates = boolean(default=False) # Save all intermediate step results
        rate_int_outliers  = boolean(default=True)  # Flag outlier pixels in rateints
        return_rateints    = boolean(default=False) # Return rateints or rate product?
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
        
        # Set additional parameters in saturation step
        self.saturation.grow_diagonal = False
        self.saturation.flag_rcsat = True

        # Initialize reference pixel correction parameters
        self.refpix.nlower = 4
        self.refpix.nupper = 4
        self.refpix.nleft = 4
        self.refpix.nright = 4
        self.refpix.nrow_off = 0
        self.refpix.ncol_off = 0

        # Ramp fit saving options
        # NOTE: `save_calibrated_ramp` is already a Detector1Pipeline property
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
            input = self.run_step(self.charge_migration, input)
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
            input = self.run_step(self.charge_migration, input)
            input = self.run_step(self.jump, input)
        
        # save the corrected ramp data, if requested
        if self.ramp_fit.save_calibrated_ramp or self.save_calibrated_ramp or self.save_intermediates:
            self.save_model(input, suffix='ramp')
        
        # Run ramp fitting & gain scale correction.
        res = self.run_step(self.ramp_fit, input, save_results=False)
        rate, rateints = (res, None) if self.ramp_fit.skip else res
        if self.rate_int_outliers and rateints is not None:
            # Flag additional outliers by comparing rateints and refit ramp
            input = self.apply_rateint_outliers(rateints, input)
            if input is None:
                input, ints_model = (rate, rateints)
            else:
                res = self.run_step(self.ramp_fit, input, save_results=False)
                input, ints_model = (res, None) if self.ramp_fit.skip else res
        else:
            # input is the rate product, ints_model is the rateints product
            input, ints_model = (rate, rateints)

        if input is None:
            self.ramp_fit.log.info('NoneType returned from ramp fitting. Gain scale correction skipped')
        else:
            self.gain_scale.suffix = 'rate'
            input = self.run_step(self.gain_scale, input, save_results=False)
        
        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            self.gain_scale.suffix = 'rateints'
            ints_model = self.run_step(self.gain_scale, ints_model, save_results=False)
            if self.save_results or self.save_intermediates:
                self.save_model(ints_model, suffix='rateints')

        # Setup output file.
        self.setup_output(input)

        if self.return_rateints:
            return ints_model
        
        return input

    def setup_output(self, input):
        """Determine output file name suffix"""
        if input is None:
            return None
        # Determine the proper file name suffix to use later
        if input.meta.cal_step.ramp_fit == 'COMPLETE':
            if self.return_rateints:
                self.suffix = 'rateints'
            else:
                self.suffix = 'rate'
        else:
            self.suffix = 'ramp'

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
        
        # Determine if we're saving results for real
        if step_obj.skip:
            # Skip save if step is skipped
            really_save_results = False
        elif (save_results is not None):
            # Use keyword specifications
            really_save_results = save_results
        elif self.save_intermediates:
            # Use save_intermediates attribute
            really_save_results = True
        elif step_obj.save_results:
            # Use step attribute
            really_save_results = True
        else:
            # Saving is unspecified
            really_save_results = False
        
        # Run step. Don't save results yet.
        step_save_orig = step_obj.save_results
        step_obj.save_results = False
        res = step_obj(input)
        step_obj.save_results = step_save_orig
        
        # Check if group scale correction or gain scale correction were skipped.
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

        from .imagetools import expand_mask
        
        # Save original step parameter.
        npix_grow = self.saturation.n_pix_grow_sat
        
        # Flag RC pixels as saturated?
        flag_rcsat = self.saturation.flag_rcsat
        if flag_rcsat:
            mask_rc = (input.pixeldq & dqflags.pixel['RC']) > 0
            # Do a bitwise OR of RC mask with groupdq to flip saturation bits
            input.groupdq = input.groupdq | (mask_rc * dqflags.pixel['SATURATED'])

        # Run step with default settings.
        if self.saturation.grow_diagonal or npix_grow == 0:
            res = self.run_step(self.saturation, input, **kwargs)
        else: # No diagonal growth
            # Initial run with 0 pixel growth
            self.saturation.n_pix_grow_sat = 0
            res = self.run_step(self.saturation, input, **kwargs)
            # Reset to original value
            self.saturation.n_pix_grow_sat = npix_grow

            self.saturation.log.info(f'Growing saturation flags by {npix_grow} pixels. Ignoring diagonal growth.')
            # Update saturation dq flags to grow in vertical and horizontal directions
            # Get saturation mask
            mask_sat = (input.groupdq & dqflags.pixel['SATURATED']) > 0

            # Expand the mask by npix_grow pixels
            mask_sat = expand_mask(mask_sat, npix_grow, grow_diagonal=False)

            # Do a bitwise OR of new mask with groupdq to flip saturation bit
            res.groupdq = res.groupdq | (mask_sat * dqflags.pixel['SATURATED'])

            # Do the same for the zero frames
            zframes = res.zeroframe if res.meta.exposure.zero_frame else None
            if zframes is not None:
                # Saturated zero frames have already been set to 0
                mask_sat = (zframes==0) | mask_rc if flag_rcsat else (zframes==0)
                # Expand the mask by npix_grow pixels
                mask_sat = expand_mask(mask_sat, npix_grow, grow_diagonal=False)
                # Set saturated pixels to 0 in zero frames
                res.zeroframe[mask_sat] = 0

        return res
    
    def apply_rateint_outliers(self, 
                               rateints_model, 
                               ramp_model, 
                               **kwargs):
        """Get pixel outliers in rateint model and apply to ramp model DQ
        
        Parameters
        ----------
        rateints_model : `~jwst.datamodels.CubeModel`
            Rateints model to use for outlier detection
        ramp_model : `~jwst.datamodels.RampModel`
            Ramp model to update with outlier flags

        Keyword Args
        ------------
        sigma_cut : float
            Sigma cut for outlier detection.
            Default is 5.
        nint_min : int
            Minimum number of integrations required for outlier detection.
            Default is 5.
        """

        from .imagetools import cube_outlier_detection

        inst = rateints_model.meta.instrument.name
        data = rateints_model.data[1:] if 'MIRI' in inst else rateints_model.data

        indbad = cube_outlier_detection(data, **kwargs)

        # Reshape outlier mask to match ramp data
        nint, ng, ny, nx = ramp_model.data.shape
        bpmask = indbad.reshape([nint, 1, ny, nx])
        if ng>1:
            bpmask = np.repeat(bpmask, ng, axis=1)

        # Update DO_NOT_USE and JUMP_DET flags
        mask_dnu = (ramp_model.groupdq & dqflags.pixel['DO_NOT_USE']) > 0
        mask_jd  = (ramp_model.groupdq & dqflags.pixel['JUMP_DET']) > 0

        # Update DO_NOT_USE and JUMP_DET flags with outlier mask
        mask_dnu = mask_dnu | bpmask
        mask_jd  = mask_jd  | bpmask

        # Update ramp model groupdq
        ramp_model.groupdq = ramp_model.groupdq | (mask_dnu * dqflags.pixel['DO_NOT_USE'])
        ramp_model.groupdq = ramp_model.groupdq | (mask_jd  * dqflags.pixel['JUMP_DET'])

        return ramp_model

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
