import os
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from jwst.stpipe import Step
from jwst import datamodels

from webbpsf_ext import robust
from webbpsf_ext.image_manip import expand_mask

class kTCSubtractStep(Step):
    """
    kTCSubtractStep: Subtract kTC noise from ramp data.
    """

    class_alias = "removektc"

    spec = """
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
    """

    def process(self, input):
        """Subtract kTC noise from data."""
        
        with datamodels.open(input) as input_model:
            datamodel = input_model.copy()
            bias_arr, _ = fit_slopes_to_ramp_data(datamodel, sat_frac=self.sat_frac)

            # Subtract bias from each integration
            nints = datamodel.meta.exposure.nints
            for i in range(nints):
                datamodel.data[i] -= bias_arr[i]

        return datamodel


class OneOverfStep(Step):
    """ OneOverfStep: Apply 1/f noise correction

    Works on `RampModel`, `CubeModel`, and `ImageModel` data products.

    For `RampModel`, it creates a master slope image, then subtracts
    a group-dependent flux image to produce a residual image. Then for
    each group residual image, we model the 1/f noise using a Savitzky-Golay 
    filter. 
    
    For `CubeModel`, we create an average of all slope images, then subtract
    from each slope image to create a series of residual images. We then apply
    a Savitzky-Golay filter to each residual image to model the 1/f noise.

    Full frame channels are handled by first removing with common channel
    noise (taking into account flips in the readout direction), then channel-dependent 
    1/f noise is modeled and applied to a complete model of the full frame image.

    If only a single integration is present, then a mask is applied to exclude
    pixels with large flux values. These pixels are excluded during the model
    fitting process.
    """

    class_alias = "1overf"

    spec = """
        model_type = option('median', 'mean', 'savgol', default='savgol') # Type of model to fit
        sat_frac = float(default=0.5) # Maximum saturation fraction for fitting
        combine_ints = boolean(default=True) # Combine integrations before ramp fitting
        vertical_corr = boolean(default=True) # Apply horizontal correction
    """

    def process(self, input):
        """Apply 1/f noise correction to a JWST `RampModel`

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be linearized.

        Returns
        =======
        output : JWST data model
            Output science data model with 1/f noise correction applied.
        """

        if isinstance(input, datamodels.RampModel):
            return self.proc_RampModel(input)
        elif isinstance(input, datamodels.CubeModel):
            return self.proc_CubeModel(input)
        elif isinstance(input, datamodels.ImageModel):
            return self.proc_ImageModel(input)
        else:
            raise ValueError(f"Do not recognize {type(input)}")

    def proc_RampModel(self, input):
        """Apply 1/f noise correction to a JWST `RampModel`

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be a linearized `RampModel`.
        """

        # Get the input data model
        with datamodels.open(input) as input_model:

            nints    = input_model.meta.exposure.nints
            ngroups  = input_model.meta.exposure.ngroups
            noutputs = input_model.meta.exposure.noutputs

            # Fit slopes to get signal
            # Grab biases for each integration and get average slope
            bias_arr, slopes = fit_slopes_to_ramp_data(input_model, 
                                                       sat_frac=self.sat_frac,
                                                       combine_ints=self.combine_ints)
            slope_mean = slopes if self.combine_ints else robust.mean(slopes, axis=0)
            # print(nints, bias_arr.shape, slope_mean.shape)
            
            # If only a single integration, then apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            if nints == 1:
                # Signal ramp should be zeros, otherwise residuals would be ~0
                signal_mean_ramp = np.zeros_like(input_model.data[0])
                good_mask = robust.mean(slope_mean, return_mask=True)
                # Expand mask by 1 pixel for good measure
                good_mask = ~expand_mask(~good_mask, 1, grow_diagonal=True)
            else:
                # Generate a mean signal ramp to subtract from each group
                group_time = input_model.meta.exposure.group_time
                ngroups = input_model.meta.exposure.ngroups
                tarr = np.arange(1, ngroups+1) * group_time
                signal_mean_ramp = slope_mean * tarr.reshape([-1,1,1])
                good_mask = np.ones_like(slope_mean, dtype=np.bool_)

            # Subtract 1/f noise from each integration
            datamodel = input_model.copy()
            data = datamodel.data
            for i in range(nints):
                cube = data[i] - bias_arr[i]
                groupdq = datamodel.groupdq[i]
                # Cumulative sum of group DQ flags
                bpmask_arr = np.cumsum(groupdq, axis=0) > 0
                for j in range(ngroups):

                    # Good pixel mask
                    im_mask = ~bpmask_arr[j] & good_mask
                    # Clean channel by channel
                    im_diff = cube[j] - signal_mean_ramp[j]

                    # Select which clean function to use
                    if noutputs>1:
                        nf_clean = CleanFullFrame(im_diff, im_mask, nout=noutputs)
                    else:
                        nf_clean = CleanSubarray(im_diff, im_mask)

                    # Perform the fit and subtract model
                    nf_clean.fit(model_type=self.model_type, vertical_corr=self.vertical_corr)
                    # if i==0 and j==ngroups//2+1:
                    #     self.quick_plot(data[i,j], nf_clean.M, nf_clean.model, im_diff=im_diff)
                    data[i,j] -= nf_clean.model

                    del nf_clean

        return datamodel
    
    def proc_CubeModel(self, input):
        """ Apply 1/f noise correction to a JWST `CubeModel` 

        Parameters
        ==========
        input : JWST data model
            Input science data model to be corrected.
            Should be a `CubeModel` such as rateints or calints.
        """
        # Get the input data model
        with datamodels.open(input) as input_model:

            nints    = input_model.meta.exposure.nints
            noutputs = input_model.meta.exposure.noutputs
            nints, ny, nx = input_model.data.shape
            chsize = nx // noutputs

            # If only a single integration, then apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            if nints==1:
                data_mean = np.zeros([ny, nx])
                good_mask = robust.mean(input_model.data, return_mask=True)
                # Expand mask by 1 pixel for good measure
                good_mask = ~expand_mask(~good_mask, 1, grow_diagonal=True)
            else:
                # Get average of data
                data = input_model.data.copy()
                for i in range(nints):
                    for ch in range(noutputs):
                        x1 = int(ch*chsize)
                        x2 = int(x1 + chsize)
                        data[i,x1:x2] -= np.nanmedian(data[i,x1:x2])
                data_mean = np.nanmean(data, axis=0)
                good_mask = np.ones_like(data_mean, dtype=np.bool_)
                del data

            # Subtract 1/f noise from each integration
            datamodel = input_model.copy()
            for i in range(nints):
                # Work on residual image
                im_diff = datamodel.data[i] - data_mean
                im_mask = (datamodel.dq[i] == 0) & good_mask

                # Select which clean function to use
                clean_func = CleanFullFrame if noutputs>1 else CleanSubarray
                nf_clean = clean_func(im_diff, im_mask)

                # Perform the fit and subtract model
                nf_clean.fit(model_type=self.model_type, vertical_corr=self.vertical_corr)
                # if i==0:
                #     self.quick_plot(datamodel.data[i], nf_clean.M, nf_clean.model, im_diff=im_diff)
                datamodel.data[i] -= nf_clean.model

                del nf_clean

        return datamodel

    def proc_ImageModel(self, input):
        """ Apply 1/f noise correction to a JWST `ImageModel` """
        # Get the input data model
        with datamodels.open(input) as input_model:

            noutputs = input_model.meta.exposure.noutputs

            # Only a single image, so apply a mask to exclude pixels
            # with a large flux values. This will be used by the model fit.
            good_mask = robust.mean(input_model.data, return_mask=True)
            # Expand mask by 1 pixel for good measure
            good_mask = ~expand_mask(~good_mask, 1, grow_diagonal=True)

            # Subtract 1/f noise from each integration
            datamodel = input_model.copy()

            # Select which clean function to use
            clean_func = CleanFullFrame if noutputs>1 else CleanSubarray
            im_mask = (datamodel.dq == 0) & good_mask
            nf_clean = clean_func(datamodel.data, im_mask)

            # Perform the fit and subtract model
            nf_clean.fit(model_type=self.model_type, vertical_corr=self.vertical_corr)
            # self.quick_plot(datamodel.data, nf_clean.M, nf_clean.model)
            datamodel.data -= nf_clean.model
            
            del nf_clean

        return datamodel
    
    def quick_plot(self, image, mask, model, im_diff=None):

        fig, axes = plt.subplots(2,2, figsize=(8,8), sharex=True, sharey=True)
        axes = axes.flatten()

        im_temp = im_diff if im_diff is not None else image
        med = np.nanmedian(im_temp)
        sig = robust.medabsdev(im_temp)
        vmin, vmax = np.array([-1,1])*sig*3 + med
        axes[0].imshow(im_temp, vmin=vmin, vmax=vmax)
        axes[1].imshow(image - model, vmin=vmin, vmax=vmax)

        # Plot mask
        axes[2].imshow(mask)

        med = np.nanmedian(model)
        sig = robust.medabsdev(model)
        vmin, vmax = np.array([-1,1])*sig*3 + med
        axes[3].imshow(model, vmin=vmin, vmax=vmax, cmap='RdBu')

        # print(np.nanmedian(im_diff), np.nanmedian(data[i,j]), np.nanmedian(nf_clean.model))
        
        fig.tight_layout()

    

def fit_slopes_to_ramp_data(input, sat_frac=0.5, combine_ints=False):
    """Fit slopes to each integration
    
    Uses custom `cube_fit` function to fit slopes to each integration.
    Returns array of slopes and bias values for each integration.
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
    combine_ints : bool
        Average all integrations into a single array before fitting.
        Return an (bias_arr, slope_mean). Default is False.
    """
    from jwst.datamodels import SaturationModel
    from jwst.saturation.saturation_step import SaturationStep
    from jwst.lib import reffile_utils
    from .utils import cube_fit

    # Get saturation reference file
    # Get the name of the saturation reference file
    sat_name = SaturationStep().get_reference_file(input, 'saturation')

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
    ny, nx = data.shape[-2:]

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
    bias_arr = np.asarray(bias_arr).reshape([nints, ny, nx])
    slope_arr = np.asarray(slope_arr).reshape([nints, ny, nx])

    # Subtract biases, average, and refit?
    if combine_ints and (nints > 1):
        data = input.data.copy()
        for i in range(nints):
            data[i] -= bias_arr[i]
            # Get group-level bpmask for this integration
            groupdq = input.groupdq[i]
            # Make sure to accumulate the group-level dq mask
            bpmask_arr = np.cumsum(groupdq, axis=0) > 0
            # Exclude bad pixels
            data[i][bpmask_arr] = np.nan

        data_mean = np.nanmean(data, axis=0)
        bias_mean = np.nanmean(bias_arr, axis=0)
        bpmask_arr = np.isnan(data_mean)
        _, slope_mean = cube_fit(tarr, data_mean, bpmask_arr=bpmask_arr,
                                 sat_vals=sat_thresh-bias_mean, sat_frac=sat_frac)
        
        # bias has shape [nints, ny, nx]
        # slope has shape [ny, nx]
        return bias_arr, slope_mean
    elif combine_ints and (nints == 1):
        # bias has shape [nints, ny, nx]
        # slope has shape [ny, nx]
        return bias_arr, slope_arr[0]
    else:
        # bias and slope arrays have shape [nints, ny, nx]
        # bias values are in units of DN and slope in DN/sec
        return bias_arr, slope_arr


class CleanFullFrame:
    """ Clean 1/f noise from full frame images

    Use `CleanSubarray` instances to clean each channel separately.
    First removes common channel noise, then cleans the residuals of 
    each channel. Final full frame model is stored in `model` attribute. 
    Intended for use on Level 1 or 2 pipeline products on any type of 
    image product (group data, rateint, calint, rate, cal, etc). 
    Suggest removing average signal levels from data and running
    this on the residuals, which better reveal the 1/f noise structure.

    Inspired by NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data. This is significantly faster than the FFT approach and provides 
    similar results. 
    """

    def __init__(self, data, mask, nout=4, exclude_outliers=True):

        # Definitions
        self.D = np.array(data, dtype=np.float32)
        self.M = np.array(mask, dtype=np.bool_)
        self.nout = nout
        self.ny, self.nx = self.D.shape
        self.chsize = self.nx // self.nout

        # Average the channel data
        self.average_channels()
        # Create a subarray class for the average channel
        chavg_mask = np.zeros_like(self.chavg, dtype=np.bool_)
        for ch in range(nout):
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            ch_mask = self.M[:,x1:x2]
            if ch % 2 == 0:
                ch_mask = np.flip(ch_mask, axis=1)
            chavg_mask = chavg_mask | ch_mask
        self.chavg_mask = chavg_mask
        # self.output_classes = {}
        self.output_classes = {
            'chavg': CleanSubarray(self.chavg, self.chavg_mask, exclude_outliers=exclude_outliers)
        }
        
        # Create subarray class for each channel
        for ch in range(self.nout):
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            self.output_classes[ch] = CleanSubarray(self.D[:,x1:x2], self.M[:,x1:x2], exclude_outliers=exclude_outliers)
    
    def average_channels(self):
        """ Create an average channel image representing the common 1/f noise"""
        data = self.D.copy()
        data[~self.M] = np.nan

        # Flip every other channel along x-axes
        for ch in range(self.nout):
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            data[:,x1:x2] -= np.nanmedian(data[:,x1:x2])
            if ch % 2 == 0:
                data[:,x1:x2] = np.flip(data[:,x1:x2], axis=1)

        # Average channels
        data = data.reshape([self.ny, self.nout, self.chsize])
        self.chavg = np.nanmean(data, axis=1)

    def fit(self, model_type='savgol', vertical_corr=False, **kwargs):

        # Fit model to average channel
        self.output_classes['chavg'].fit(model_type=model_type, vertical_corr=False, **kwargs)
        chavg_model = self.output_classes['chavg'].model

        # Get final models
        final_model = np.zeros_like(self.D)
        for ch in range(self.nout):
            ch_class = self.output_classes[ch]

            # Flip model every other channel
            avgmod = np.flip(chavg_model, axis=1) if ch % 2 == 0 else chavg_model

            # Subtract average channel model from data
            ch_class.D -= avgmod
            ch_class.fit(model_type=model_type, vertical_corr=vertical_corr, **kwargs)

            # Add model back to data
            x1 = int(ch*self.chsize)
            x2 = int(x1 + self.chsize)
            final_model[:,x1:x2] = ch_class.model + avgmod

        self.model = final_model

    def clean(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Clean the data

        Overwrites data in-place with the cleaned data.
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """ 
        # Fit the background model
        self.fit(model_type=model_type, vertical_corr=vertical_corr, **kwargs) 
        self.D -= self.model # Overwrite data with cleaned data
        return self.D


class CleanSubarray:
    """ 1/f noise modeling and subtraction for HAWAII-2RG subarrays.

    CleanSubarray is the base class for removing residual correlated
    read noise from generic JWST near-IR Subarray images.  It is
    intended for use on Level 1 or 2 pipeline products on any type of
    image product (group data, rateint, calint, rate, cal, etc). 
    Suggest removing average signal levels from data and running
    this on the residuals, which better reveal the 1/f noise structure.

    Inspired by NSClean by Bernie Rauscher (https://arxiv.org/abs/2306.03250),
    however instead of using FFTs and Matrix multiplication, this class uses
    Savitzky-Golay filtering to model the 1/f noise and subtract it from the
    data. This is significantly faster than the FFT approach and provides 
    similar results. 

    This function assumes such that the slow-scan runs vertically along the
    y-axis, while the fast scan direction runs horizontally. Direction 
    (left to right or right to left) is irrelevant.
    """
            
    def __init__(self, data, mask, exclude_outliers=True):
        """ Initialize the class

        Parameters
        ==========
        data : ndarray
            Two-dimensional input data.
        mask : ndarray
            Two dimensional background pixels mask. 
            Pixels = True are modeled as background.
            Pixels = False are excluded from the background model.
        exclude_outliers : bool
            Exclude statistical outliers and their nearest neighbors
            from the background pixels mask.
        """

        # Definitions
        self.D = np.array(data, dtype=np.float32)
        self.M = np.array(mask, dtype=np.bool_)
        
        # The mask potentially contains NaNs. Exclude them.
        self.M[np.isnan(self.D)] = False
        
        # The mask potentially contains statistical outliers.
        # Optionally exclude them.
        if exclude_outliers is True:
            gdpx = robust.mean(self.D, Cut=3, return_mask=True)
            bdpx = ~gdpx # Bad pixel mask
            bdpx = expand_mask(bdpx, 1, grow_diagonal=False) # Also flag 4 nearest neighbors
            gdpx = ~bdpx # Good pixel mask
            self.M = self.M & gdpx

        # Median subtract
        self.D = self.D - np.nanmedian(self.D[self.M])

    @property
    def nx(self):
        """ Number of columns in the image """
        return self.D.shape[1]
    @property
    def ny(self):
        """ Number of rows in the image """
        return self.D.shape[0]

    def fit(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Return the model which is just median of each row
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a vertical correction to the data. This is useful
            for removing vertical striping. The default is False.
        """

        # Fit the model
        if 'median' in model_type:
            self.model = self._fit_median(robust_mean=False)
        elif 'mean' in model_type:
            self.model = self._fit_median(robust_mean=True)
        elif 'savgol' in model_type:
            self.model = self._fit_savgol(**kwargs)
        else:
            raise ValueError(f"Do not recognize model_type={model_type}")
        
        # Apply horizontal correction
        if vertical_corr:
            data = self.D.copy()
            mask = self.M.copy()
            model_vert = self.model.copy()
            # Rotate model subtracted data by 90 degrees
            self.D = np.rot90(data - model_vert, k=1, axes=(0,1))
            self.M = np.rot90(self.M, k=1, axes=(0,1))
            # Fit the model
            self.fit(model_type=model_type, vertical_corr=False, **kwargs)
            
            # Return data and mask to original orientation
            self.D = data
            # self.M = mask
            self.M = np.rot90(self.M, k=-1, axes=(0,1))
            self.model = np.rot90(self.model, k=-1, axes=(0,1)) + model_vert

            del model_vert

    def _fit_median(self, robust_mean=False):
        """ Return the model which is just median of each row
        
        Option to use robust mean instead of median.
        """

        mean_func = robust.mean if robust_mean else np.nanmedian

        # Fit the model
        data = self.D.copy()
        data[~self.M] = np.nan
        return mean_func(data, axis=1).repeat(self.nx).reshape(data.shape)
        
    def _fit_savgol(self, niter=5, **kwargs):
        """ Use a Savitzky-Golay filter to smooth the masked row data

        Parameters
        ==========
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.

        Keyword Args
        ============
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """

        bpmask = ~self.M # Bad pixel mask
        model = channel_smooth_savgol(self.D, mask=bpmask, **kwargs)

        for i in range(niter):
            # Get median model
            model_med = self._fit_median(robust_mean=False)

            # Find standard deviation of difference between model and median values
            bpmask = ~self.M 
            diff = model - model_med
            diff[bpmask] = np.nan
            sig = np.nanstd(diff, ddof=1)

            # Flag new outliers
            bp_sig = np.abs(diff) > 3*sig
            bpmask_new = bpmask | bp_sig
            if bpmask_new.sum() == bpmask.sum():
                break

            # Update mask and refit model
            self.M = ~bpmask_new
            model = channel_smooth_savgol(self.D, mask=bpmask_new, **kwargs)

        return model
        
    def clean(self, model_type='savgol', vertical_corr=False, **kwargs):
        """ Clean the data

        Overwrites data in-place with the cleaned data.
        
        Parameters
        ==========
        model_type : str
            Must be 'median', 'mean', or 'savgol'. For 'mean' case,
            it uses a robust mean that ignores outliers and NaNs.
            The 'median' case uses `np.nanmedian`. The 'savgol' case
            uses a Savitzky-Golay filter to model the 1/f noise, 
            iteratively rejecting outliers from the model fit relative
            to the median model. The default is 'savgol'.
        vertical_corr : bool
            Apply a horizontal correction to the data. This is useful
            for removing horizontal striping. The default is False.

        Keyword Args
        ============
        niter : int
            Number of iterations to use for rejecting outliers during
            the model fit. If the number of rejected pixels does not
            change between iterations, then the fit is considered
            converged and the loop is broken.
        winsize : int
            Size of the window filter. Should be an odd number.
        order : int
            Order of the polynomial used to fit the samples.
        per_line : bool
            Smooth each channel line separately with the hopes of avoiding
            edge discontinuities.
        mask : bool image or None
            An image mask of pixels to ignore. Should be same size as im_arr.
            This can be used to mask pixels that the filter should ignore, 
            such as stellar sources or pixel outliers. A value of True indicates
            that pixel should be ignored.
        mode : str
            Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
            determines the type of extension to use for the padded signal to
            which the filter is applied.  When `mode` is 'constant', the padding
            value is given by `cval`. 
            When the 'interp' mode is selected (the default), no extension
            is used.  Instead, a degree `polyorder` polynomial is fit to the
            last `window_length` values of the edges, and this polynomial is
            used to evaluate the last `window_length // 2` output values.
        cval : float
            Value to fill past the edges of the input if `mode` is 'constant'.
            Default is 0.0.
        """ 
        # Fit the background model
        self.fit(model_type=model_type, vertical_corr=vertical_corr, **kwargs) 
        self.D -= self.model # Overwrite data with cleaned data
        return self.D

def mask_helper():
    """Helper to handle indices and logical indices of a mask

    Output: index, a function, with signature indices = index(logical_indices),
    to convert logical indices of a mask to 'equivalent' indices

    Example:
        >>> # linear interpolation of NaNs
        >>> mask = np.isnan(y)
        >>> x = mask_helper(y)
        >>> y[mask]= np.interp(x(mask), x(~mask), y[~mask])
    """
    return lambda z: np.nonzero(z)[0]

def channel_smooth_savgol(im_arr, winsize=127, order=3, per_line=False, 
    mask=None, **kwargs):
    """Channel smoothing using savgol filter

    Function for generating a map of the 1/f noise within a series of input images.

    Parameters
    ==========
    im_arr : ndarray
        Input array of images (intended to be a cube of output channels).
        Shape should either be (ny, chsize) to smooth a single channel or
        (nchan, ny, chsize) for  multiple channels.
        Each image is operated on separately. If only two dimensions,
        then only a single input image is assumed. NaN's will be
        interpolated over.

    Keyword Args
    ============
    winsize : int
        Size of the window filter. Should be an odd number.
    order : int
        Order of the polynomial used to fit the samples.
    per_line : bool
        Smooth each channel line separately with the hopes of avoiding
        edge discontinuities.
    mask : bool image or None
        An image mask of pixels to ignore. Should be same size as im_arr.
        This can be used to mask pixels that the filter should ignore, 
        such as stellar sources or pixel outliers. A value of True indicates
        that pixel should be ignored.
    mode : str
        Must be 'mirror', 'constant', 'nearest', 'wrap' or 'interp'.  This
        determines the type of extension to use for the padded signal to
        which the filter is applied.  When `mode` is 'constant', the padding
        value is given by `cval`. 
        When the 'interp' mode is selected (the default), no extension
        is used.  Instead, a degree `polyorder` polynomial is fit to the
        last `window_length` values of the edges, and this polynomial is
        used to evaluate the last `window_length // 2` output values.
    cval : float
        Value to fill past the edges of the input if `mode` is 'constant'.
        Default is 0.0.
    """
    from scipy.signal import savgol_filter

    sh = im_arr.shape
    if len(sh)==2:
        nz = 1
        ny, nx = sh
    else:
        nz, ny, nx = sh

    # Check that winsize is odd
    winsize = winsize-1 if winsize % 2 == 0 else winsize

    # Reshape in case of nz=1
    im_arr = im_arr.reshape([nz, -1])
    if mask is not None:
        mask = mask.reshape([nz, -1])

    res_arr = []
    for i, im in enumerate(im_arr):
        # im should be a 1D array

        # Interpolate over masked data and NaN's
        nans = np.isnan(im)
        im_mask = nans if mask is None else nans | mask[i].flatten()
        if im_mask.any():
            # Create a copy so as to not change the original data
            im = np.copy(im)

            # Use a savgol filter to smooth out any outliers
            res = im.copy()
            res[~im_mask] = savgol_filter(im[~im_mask], 33, 3, mode='interp')

            # Replace masked pixels with linear interpolation
            x = mask_helper() # Returns the nonzero (True) indices of a mask
            im[im_mask]= np.interp(x(im_mask), x(~im_mask), res[~im_mask])

        if per_line:
            im = im.reshape([ny,-1])

            res = savgol_filter(im, winsize, order, axis=1, delta=1, **kwargs)
            res_arr.append(res)
        else:
            res = savgol_filter(im, winsize, order, delta=1, **kwargs)
            res_arr.append(res.reshape([ny,-1]))
    res_arr = np.asarray(res_arr)

    if len(sh)==2:
        return res_arr[0]
    else:
        return res_arr
